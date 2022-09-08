import argparse

class BaseOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser()

        # data directory
        self.parser.add_argument("-i", "--input", required=False, help="Path to input .types file")
        self.parser.add_argument("-d", "--dir", required=False, help="Path to directory with proteins and ligand structures")
        self.parser.add_argument("-o", "--output_dir", required=False, help="Name of directory to save models")
        self.parser.add_argument("-opath", "--output_path", type=str, default="/c7/home/margonza/Documents/cnns/models/shapeNW/shapeNN", help="Path to dir to save models.")
        self.parser.add_argument("-oplpath", "--output_path_glp", required=False, help="Dir to save generated ligand and proteins")
        self.parser.add_argument('--shuffle', action='store_true', help="To not shuffle data don't include --shuffle in args")
        self.parser.add_argument("--grid_resol", type=int, default=1.0, help="Grid resolution")
        self.parser.add_argument("--grid_dim", type=int, default=23.0, help="Grid dimension")
        self.parser.add_argument("--translation", type=float, default=0.2, help="Grid translation")
        self.parser.add_argument('--rotation', action='store_true', help="To not rotate data don't include --rotation in args")

        # hyperparameters
        self.parser.add_argument("--num_epoch", type=int, default=101, help="number of epochs")
        self.parser.add_argument("--num_down", type=int, default=8, help="number of layers in Generator")
        self.parser.add_argument("--batch_size", type=int, default=6, help="input batch size")
        self.parser.add_argument("--nz", type=int, default=8, help="latent vector size")
        self.parser.add_argument("--num_Ds", type=int, default=2, help="number of Discrminators")
        self.parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
        self.parser.add_argument("--niter_decay", type=int, default=50, help="# of iter to linearly decay learning rate to zero")
        self.parser.add_argument("--decay_rate", type=int, default=0.95, help="decay rate")

        # loss function weights
        self.parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
        self.parser.add_argument("--beta2", type=float, default=0.999, help="B2 term of adam")
        self.parser.add_argument("--lambda_L1", type=float, default=10.0, help="weight for |B-G(A, E(B))|")
        self.parser.add_argument("--lambda_z", type=float, default=0.5, help="weight for ||E(G(random_z)) - random_z||")
        self.parser.add_argument("--lambda_kl", type=float, default=0.01, help="weight for KL loss")
        self.parser.add_argument("--lambda_VAEGAN", type=float, default=1.0, help="weight for adversarial VAE-GAN")
        self.parser.add_argument("--lambda_LRGAN", type=float, default=1.0, help="weight for adversarial LR-GAN")
        self.parser.add_argument("--lambda_VAE_disc", type=float, default=1.0, help="weight for VAE-GAN discriminators")
        self.parser.add_argument("--lambda_LRGAN_disc", type=float, default=1.0, help="weight for LR-GAN discriminators")

        # GPU ids
        self.parser.add_argument("--GPUIDS", required=True, nargs="+", type=int, help="GPU ids available, example 0 1 2")

    def create_parser(self):
        return vars(self.parser.parse_args())

args = BaseOptions().create_parser()

print(args)
