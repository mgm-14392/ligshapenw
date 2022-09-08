import model as model_2
import molgrid
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
from save_density import save_density
import math

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# initialize network
generator = model_2.G_Unet_add_all3D(nz = 8, norm_layer = nn.InstanceNorm3d,
                                     nl_layer = nn.ReLU, use_dropout = False, gpu_ids = 0,
                                     num_downs = 8).to(device)

# Setup optimizers for networks
optimizer_G = torch.optim.Adam(generator.parameters(),
                               lr = 0.0002, betas=(0.5, 0.999))

# load network checkpoints
gen_checkpoint = torch.load('generator_6270.pth')

generator.load_state_dict(gen_checkpoint['model_state_dict'])
optimizer_G.load_state_dict(gen_checkpoint['optimizer_state_dict'])
nn.DataParallel(generator)
generator.eval()


# ----------
#  Load data
# ----------

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

# options and inputs
datadir = 'cnns/curated_shapeNW_data'
dname='eval_P06401.types'

# Each example is a line in the DUDE.types files
# If you have a receptor and ligand and 14 atom type channels, each example will have 28 channels.
d = molgrid.ExampleProvider(shuffle= False, balanced=False,
                            stratify_receptor=False, labelpos=0, stratify_pos=1, stratify_abs=False, stratify_min=0,
                            stratify_max=0, stratify_step=0, group_batch_size=1, max_group_size=0,
                            cache_structs=True, add_hydrogens=False, duplicate_first=False, num_copies=1,
                            make_vector_types=False, data_root=datadir, recmolcache="", ligmolcache="")

d.populate(dname)

# Total prots
dataset_size = d.size()

box_size = 24
# initialize libmolgrid GridMaker,
gmaker = molgrid.GridMaker(resolution = 1.0, dimension = (box_size-1), binary=False,
                           radius_type_indexed=False, radius_scale=1.0, gaussian_radius_multiple=1.0)

# d.num_types is the number atom type channels for an example
ddims = gmaker.grid_dimensions(d.num_types())
print(ddims)
dtensor_shape = (1,) + ddims
dinput_tensor = torch.zeros(dtensor_shape, dtype=torch.float32).to(device)
print(dinput_tensor.shape)

channels = {1:'AliphaticCHydrophobe',2:'AliphaticCNonHydrophobe',3:'AromaticCHydrophobe',4:'AromaticCNonHydrophobe',5:'Br_I',6:'Cl',7:'F',8:'N_NAcceptor',9:'NDonor_NDonorAcceptor',10:'O_OAcceptor',11:'O_DonorAcceptor_ODonor',12:'S_SAcceptor',13:'P',14:'Metal',15:'void'}
proteins = {1:"",2:"",3:"",4:""}

number_gen_ligs = 100
batch_size = 1
n_iter = math.ceil(dataset_size/batch_size)

for itr in range(1, n_iter+1):
    # each batch has the concatenated protein-ligand pairs
    dbatch = d.next_batch(batch_size)
    center = list(dbatch[0].coord_sets[-1].center())
    print(center)
    origin = np.asarray(center) - box_size / 2.
    print(origin)
    gmaker.forward(dbatch, dinput_tensor, random_translation = 0.0, random_rotation = False)

    # split tensor by channels, it has 28 channels first 14 are for the protein and second 14 for the ligand
    chunks = dinput_tensor.chunk(chunks=2, dim=1)

    real_data = {'p': normVoid_tensor(chunks[0]),
                 'l': normVoid_tensor(chunks[1])
                 }

    for i in range(1,number_gen_ligs+1):

        sampled_z = torch.randn(real_data['p'].size(0), 8).to(device)
        print(sampled_z.shape)

        print(real_data['p'].shape)
        fake_lig_gen = generator(real_data['p'], sampled_z)
        torch.save(fake_lig_gen, 'gen_lig_%d_%d.pt' % (i,itr))

        for batch in range(fake_lig_gen.shape[0]):
            for channel in range(fake_lig_gen.shape[1]):
                ch = fake_lig_gen[batch][channel].cpu().detach().numpy()
                out = "genlig_%s_%d_%d.mrc" % (channels[channel+1], i, itr)
                save_density(ch, out, 1.0, origin, 0)

                if i == 0:
                    p = real_data['p'][batch][channel].cpu().numpy()
                    out = "p_%s.mrc" % channels[channel+1]
                    save_density(p, out, 1.0, origin, 0)

                    l = real_data['l'][batch][channel].cpu().numpy()
                    out = "l_%s.mrc" % channels[channel+1]
                    save_density(l, out, 1.0, origin, 0)

