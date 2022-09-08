import os
import model as model_2
import baseoptions
import molgrid
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from torchvision import models
import torch.nn.parallel
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import sys
import math
import datetime
import time
import numpy as np

args = baseoptions.BaseOptions().create_parser()

batch_size = args['batch_size']

# Directory to save models
new_d = args['output_dir']
save = args['output_path']
path = os.path.join(save,new_d)
os.makedirs(path, exist_ok = True)

# Directory to save ligs and generated ligs
gen_ligs_dir = args['output_path_glp']

#tensorboard to visualize loss functions
tb = SummaryWriter()

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# weight initialization
def weights_init(m):
    if isinstance(m, nn.Conv3d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)


# initialize network
generator = model_2.G_Unet_add_all3D(nz = args['nz'], norm_layer = nn.InstanceNorm3d,
                                     nl_layer = nn.ReLU, use_dropout = False,
                                     gpu_ids = args['GPUIDS'], num_downs = args['num_down'])

encoder = model_2.E_3DNLayers(norm_layer = nn.InstanceNorm3d, nl_layer = nn.ReLU,
                              vaeLike = True, gpu_ids = args['GPUIDS'])

D_cVAE = model_2.D_N3Dv1LayersMulti(input_nc = 30, norm_layer = nn.InstanceNorm3d,
                                    use_sigmoid = False, gpu_ids = args['GPUIDS'],
                                    num_D = args['num_Ds'])

D_cLR = model_2.D_N3Dv1LayersMulti(input_nc = 30, norm_layer = nn.InstanceNorm3d,
                                   use_sigmoid = False, gpu_ids = args['GPUIDS'],
                                   num_D = args['num_Ds'])

models = [generator, encoder, D_cVAE, D_cLR]

# initialize wieghts, move to GPU or keep in CPU
for model in models:
    if torch.cuda.device_count() > 1:
        print('number of CUDA devices: %d . Initializing model in training mode'  % torch.cuda.device_count())
        model.train()
        model.apply(weights_init)
        model = nn.DataParallel(model)
        model.to(device)
    else:
        print('number of CUDA devices: %d . Initializing model in training mode' % torch.cuda.device_count())
        model.train()
        model.apply(weights_init)
        model.to(device)

# ----------
#  Loss functions
# ----------

def pearson_Corr(output_tensor,target_tensor):
    voutput_tensor = output_tensor - torch.mean(output_tensor)
    vtarget_tensor = target_tensor - torch.mean(target_tensor)
    cost = torch.sum(voutput_tensor * vtarget_tensor) / (torch.sqrt(torch.sum(voutput_tensor ** 2)) * torch.sqrt(torch.sum(vtarget_tensor ** 2)))
    return cost

def channel_pearson_corr(output_tensor,target_tensor):
    # mean by channel and reshape to do substraction
    voutput_tensor = output_tensor - output_tensor.mean(dim = [2,3,4]).view(output_tensor.shape[0], output_tensor.shape[1], 1, 1, 1)
    vtarget_tensor = target_tensor - target_tensor.mean(dim = [2,3,4]).view(target_tensor.shape[0], target_tensor.shape[1], 1, 1, 1)
    # pearson correlation between each channels but for all elements in the batch. cost.shape would be = [14]
    pc = torch.sum((voutput_tensor * vtarget_tensor),dim = [0,2,3,4]) / (torch.sqrt(torch.sum((voutput_tensor ** 2),dim = [0,2,3,4])) * torch.sqrt(torch.sum((vtarget_tensor ** 2), dim = [0,2,3,4])))
    # pearson correlation between each channels for each element in batch change dim to [2,3,4]. cost.shape would be = [batch_size,14]
    return pc

def channel_MSE(output_tensor,target_tensor):
    ch_MSE = torch.mean(((target_tensor - output_tensor)**2), dim=[0,2,3,4])
    return ch_MSE

def channel_EUCLIDEAN(output_tensor,target_tensor):
    ch_euc = torch.sqrt(torch.sum(((target_tensor - output_tensor)**2), dim=[0,2,3,4]))
    return ch_euc

def channel_L1(output_tensor,target_tensor):
    ch_l1 = torch.mean(torch.abs(target_tensor - output_tensor), dim=[0,2,3,4])
    return ch_l1

def check_channel_values(norm_tensor):
    return norm_tensor.sum(dim=1)

def cross_entropy(real_tensor, fake_tensor, eps=1e-5):
    CE = -1*((real_tensor * torch.log(fake_tensor + eps)).sum(dim=1).mean())
    return CE

# Initialize Loss functions LSGAN uses MSE loss
mae_loss = torch.nn.L1Loss().to(device)
L1_loss = torch.nn.L1Loss().to(device)
MSE_loss = torch.nn.MSELoss().to(device)

# Setup optimizers for networks
optimizer_E = torch.optim.Adam(encoder.parameters(), lr = args['lr'],
                               betas=(args['beta1'], args['beta2']))

optimizer_G = torch.optim.Adam(generator.parameters(), lr = args['lr'],
                               betas=(args['beta1'], args['beta2']))

optimizer_cDVAE = torch.optim.Adam(D_cVAE.parameters(), lr = args['lr'],
                                   betas=(args['beta1'], args['beta2']))

optimizer_cDLR = torch.optim.Adam(D_cLR.parameters(), lr = args['lr'],
                                  betas=(args['beta1'], args['beta2']))

optimizer_list = [optimizer_E, optimizer_G, optimizer_cDVAE, optimizer_cDLR]

# ----------
#  Load data
# ----------

# options and inputs
datadir = args['dir']
dname = args['input']

# Normalize tensor values
def normVoid_tensor(input_tensor):
    # normalize
    input_tensor /= input_tensor.sum(axis = 1).max()
    # add void dimension
    input_tensor = torch.cat((input_tensor, (1 - input_tensor.sum(dim = 1).unsqueeze(1))), axis=1)
    # clamp values
    input_tensor = torch.clamp(input_tensor, min=0, max=1)
    # normalize again adds to one
    input_tensor /= input_tensor.sum(axis = 1).unsqueeze(1)
    return input_tensor

# Each example is a line in the DUDE.types files
# If you have a receptor and ligand and 14 atom type channels, each example will have 28 channels.
d = molgrid.ExampleProvider(shuffle=args['shuffle'], balanced=False,
                            stratify_receptor=False, labelpos=0, stratify_pos=1, stratify_abs=False, stratify_min=0,
                            stratify_max=0, stratify_step=0, group_batch_size=1, max_group_size=0,
                            cache_structs=True, add_hydrogens=False, duplicate_first=False, num_copies=1,
                            make_vector_types=False, data_root=datadir, recmolcache="", ligmolcache="")

d.populate(dname)

# Total prots
dataset_size = d.size()

# initialize libmolgrid GridMaker,
# specify grid resolution; dimension along each side of the cube;
gmaker = molgrid.GridMaker(resolution = args['grid_resol'], dimension = args['grid_dim'], binary=False,
                           radius_type_indexed=False, radius_scale=1.0, gaussian_radius_multiple=1.0)

# d.num_types is the number atom type channels for an example
ddims = gmaker.grid_dimensions(d.num_types())
dtensor_shape = (batch_size,) + ddims
dinput_tensor = torch.zeros(dtensor_shape, dtype=torch.float32).to(device)

# ------------
# Training
# ------------

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

def reparametrize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = torch.cuda.FloatTensor(std.size()).normal_()
    eps = Variable(eps)
    return eps.mul(std).add_(mu)

# Number of training epochs
num_epochs = args['num_epoch']
# number iterations
n_iter = math.ceil(dataset_size/batch_size)

prev_time = time.time()
begin_time = datetime.datetime.now()

lr = args['lr']
i = 0
for epoch in range(0, num_epochs):

    lv = 0
    pearson_lv = 0
    lz = 0
    lk = 0
    ldvae = 0
    ldlr = 0
    legDVAE = 0
    legDLR = 0
    leg = 0

    for itr in range(0, n_iter):

        # each batch has the concatenated protein-ligand pairs
        dbatch = d.next_batch(batch_size)
        gmaker.forward(dbatch, dinput_tensor, random_translation = args['translation'], random_rotation = args['rotation'])

        # split tensor by channles, it has 28 channels first 14 are for the protein and second 14 for the ligand
        chunks = dinput_tensor.chunk(chunks=2, dim=1)

        if epoch == 2 and itr == 2:
            torch.save(chunks[0], '/c7/scratch2/margonza/Documents/cnns/scripts/scripts/shapenetwork/%s/prot_nonorm_%d_%d.pt' % (gen_ligs_dir, epoch, itr))
            torch.save(chunks[1], '/c7/scratch2/margonza/Documents/cnns/scripts/scripts/shapenetwork/%s/lig_nomor_%d_%d.pt' % (gen_ligs_dir, epoch, itr))

        real_data = {'p': normVoid_tensor(chunks[0]),
                     'l': normVoid_tensor(chunks[1])}

        #------------------
        # Train  Generator and Encoder
        #------------------

        optimizer_E.zero_grad()
        optimizer_G.zero_grad()

        #-----------------
        #  cVAE-GAN
        #-----------------

        # Encode the real ligand
        mu, log_variance = encoder(real_data['l'])
        encoded_z = reparametrize(mu, log_variance)
        # Generate fake ligands  with encoded ligand
        fake_lig_cVAE = generator(real_data['p'], encoded_z)
        # reconstruction of the ligand
        loss_voxel = cross_entropy(real_data['l'], fake_lig_cVAE).to(device)
        lv += loss_voxel.item() * args['lambda_L1']

        # KL divergence for encoder distribution
        loss_kl = 0.5 * torch.sum(torch.exp(log_variance) + mu ** 2 - log_variance -1 )
        lk += loss_kl.item() * args['lambda_kl']

        # Adversarial loss
        fake_pair_cVAE = torch.cat([ real_data['p'], fake_lig_cVAE], 1)
        D_VAE_GAN = D_cVAE(fake_pair_cVAE)
        loss_VAE_GAN = sum([MSE_loss(out, torch.ones_like(out)) for out in D_VAE_GAN])
        legDVAE += loss_VAE_GAN.item()

        # other loss implementations to monitor progress in ligand reconstruction
        L1_LIG = L1_loss(fake_lig_cVAE, real_data['l'])

        pearson_correlation = pearson_Corr(fake_lig_cVAE, real_data['l']).item()
        pearson_lv += pearson_correlation

        PC = channel_pearson_corr(fake_lig_cVAE, real_data['l'])
        print(PC)

        ch_MSE = channel_MSE(fake_lig_cVAE, real_data['l'])
        print(ch_MSE)

        ch_euc = channel_EUCLIDEAN(fake_lig_cVAE, real_data['l'])
        print(ch_euc)

        ch_l1 = channel_L1(fake_lig_cVAE, real_data['l'])
        print(ch_l1)

        #---------------
        # cLR-GAN
        #---------------

        # D_cLR-GAN has to discriminate [pl, pl] vs [p, G(p, random_z)]
        sampled_z = Variable(Tensor(np.random.normal(0, 1, (real_data['p'].size(0), args['nz']))))
        fake_lig_LRGAN = generator(real_data['p'], sampled_z)

        # adversarial loss
        fake_pair_LRGAN = torch.cat([ real_data['p'], fake_lig_LRGAN], 1)
        LR_GAN = D_cLR(fake_pair_LRGAN)
        loss_LR_GAN = sum([MSE_loss(out, torch.ones_like(out)) for out in LR_GAN])
        legDLR += loss_LR_GAN.item()

        #---------------
        # Total loss (Generator + Encoder)
        #---------------

        loss_GE = loss_VAE_GAN * args['lambda_VAEGAN'] + loss_LR_GAN * args['lambda_LRGAN'] + loss_voxel * args['lambda_L1'] + loss_kl * args['lambda_kl']
        leg += loss_GE.item()

        loss_GE.backward(retain_graph = True)
        optimizer_E.step()

        #----------------
        # Generator only loss
        #----------------

        # Latent L1 loss
        _mu, _ = encoder(fake_lig_LRGAN)
        loss_latent = mae_loss(_mu, sampled_z) * args['lambda_z']
        lz += loss_latent * 0.5

        loss_latent.backward()
        optimizer_G.step()

        #---------------
        # Train Discriminator (cVAE-GAN)
        #----------------

        optimizer_cDVAE.zero_grad()
        D_VAE_real = D_cVAE(torch.cat([ real_data['p'], real_data['l']], 1))
        D_VAE_fake = D_cVAE(fake_pair_cVAE.detach())

        # discriminator loss
        D_VAEreal_loss = sum([MSE_loss(out, torch.ones_like(out)) for out in D_VAE_real])
        D_VAEfake_loss = sum([MSE_loss(out, torch.zeros_like(out)) for out in D_VAE_fake])

        loss_D_VAE = D_VAEreal_loss * args['lambda_VAE_disc']  + D_VAEfake_loss * args['lambda_VAE_disc']
        ldvae += loss_D_VAE.item()

        loss_D_VAE.backward()
        optimizer_cDVAE.step()

        #------------------
        # train Discriminator (cLR-GAN)
        #------------------

        optimizer_cDLR.zero_grad()
        loss_DLR_real = D_cLR(torch.cat([ real_data['p'], real_data['l']], 1))

        loss_DLR_fake = D_cLR(fake_pair_LRGAN.detach())

        # discriminator loss
        D_LRreal_loss = sum([MSE_loss(out, torch.ones_like(out)) for out in loss_DLR_real])
        D_LRfake_loss = sum([MSE_loss(out, torch.zeros_like(out)) for out in loss_DLR_fake])

        loss_D_LR = D_LRreal_loss * args['lambda_LRGAN_disc'] + D_LRfake_loss * args['lambda_LRGAN_disc']
        ldlr += loss_D_LR.item()

        loss_D_LR.backward()
        optimizer_cDLR.step()

        #--------------
        # log progress
        #---------------

        # Determine time left
        total_batches = n_iter * num_epochs
        batches_done = epoch * n_iter
        batches_left = total_batches - batches_done
        time_left = datetime.timedelta(seconds = batches_left * (time.time()- prev_time))
        prev_time = time.time()

        #---------------
        # save last generated ligands
        #---------------
        if epoch % 50 == 0:
            if itr % 100 == 0:
                torch.save(fake_lig_cVAE, '/%s/generated_LR_VAE_%d_%d.pt' % (gen_ligs_dir, epoch, itr))
                torch.save(fake_lig_LRGAN, '/%s/generated_LR_LRGAN_%d_%d.pt' % (gen_ligs_dir, epoch, itr))
                torch.save(real_data['l'], '/%s/lig_LR_%d_%d.pt' % (gen_ligs_dir, epoch, itr))
                torch.save(real_data['p'], '/%s/prot_LR_%d_%d.pt' % (gen_ligs_dir, epoch, itr))

        sys.stdout.write("\r Epoch %d %d Batch %d %d D_VAE_loss %f D_LR_loss %f GE loss %f "
                         "GDVAE %f GDLRGAN %f recon_voxel %f pc %f kl %f latent %f ETA: %s \n"
                         %(
                             epoch,
                             num_epochs,
                             itr,
                             n_iter,
                             (loss_D_VAE.item() / batch_size),
                             (loss_D_LR.item() / batch_size),
                             (loss_GE.item() / batch_size),
                             (loss_VAE_GAN.item() / batch_size),
                             (loss_LR_GAN.item() / batch_size),
                             (loss_voxel.item() / batch_size),
                             (pearson_correlation / batch_size),
                             (loss_kl.item() / batch_size),
                             (loss_latent.item() / batch_size),
                             time_left,
                         )
                         )

    # average values
    av_lv = (lv/n_iter) / batch_size
    av_pearson_lv = (pearson_lv/n_iter) / batch_size
    av_legDVAE = (legDVAE/n_iter) / batch_size
    av_legDLR = (legDLR/n_iter) / batch_size
    av_leg = (leg/n_iter) / batch_size
    av_lz = (lz/n_iter) /batch_size
    av_ldvae = (ldvae/n_iter) / batch_size
    av_ldlr = (ldlr/n_iter) / batch_size


    # -------------
    # start LR scheduler
    # --------------

    #if epoch == 20:
    #    print('Epoch:', epoch,'old LR:', lr)
    #    lr = lr * 0.1
    #    i+=1
    #    for optimizer in optimizer_list:
    #        for g in optimizer.param_groups:
    #            g['lr'] = lr
    #            print('Epoch:', epoch,'new LR:', lr)

    # -----------
    # tensorboard
    # -----------

    tb.add_scalar("Ligand reconstruction perf = 0", av_lv, epoch)
    tb.add_scalar("Ligand pearsonCorr perf = 1,-1", av_pearson_lv, epoch)
    tb.add_scalar("Encoder-Generator DVAE perf = 0", av_legDVAE, epoch)
    tb.add_scalar("Encoder-Generator DLGAN perf = 0", av_legDLR, epoch)
    tb.add_scalar("Encoder Generator perf = 0", av_leg, epoch)
    tb.add_scalar("Generator Z recon perf = 0", av_lz, epoch)
    tb.add_scalar("D_VAE loss Disc = 0", av_ldvae, epoch)
    tb.add_scalar("D_LR loss Disc = 0", av_ldlr, epoch)

    if epoch  % 30 == 0:
        torch.save({'epoch':epoch, 'model_state_dict':generator.state_dict(), 'optimizer_state_dict':optimizer_G.state_dict()}, path + "/generator_%d.pth" % epoch)
        torch.save({'epoch':epoch, 'model_state_dict':encoder.state_dict(), 'optimizer_state_dict':optimizer_E.state_dict()}, path + "/encoder_%d.pth" % epoch)
        torch.save({'epoch':epoch, 'model_state_dict':D_cVAE.state_dict(), 'optimizer_state_dict':optimizer_cDVAE.state_dict()}, path + "/D_VAE_%d.pth" % epoch)
        torch.save({'epoch':epoch, 'model_state_dict':D_cLR.state_dict(), 'optimizer_state_dict':optimizer_cDLR.state_dict()}, path + "/D_LR_%d.pth" % epoch)
