# # The netwokrs are based on BycycleGAN (https://github.com/junyanz/BicycleGAN) and
# # Ligdream (https://github.com/compsciencelab/ligdream)
# # code from: https://doi.org/10.1021/acs.molpharmaceut.9b00634 all credit to authors
# # Networks are written in pytorch==0.3.1

import torch
import torch.nn as nn
import torch.utils.data

#### bicyclegan - shape generation

class ListModule(object):
    # should work with all kind of module
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(
                self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


class D_N3Dv1LayersMulti(nn.Module):
    """
    Disriminator network of 3D-BicycleGAN
    """
    def __init__(self, input_nc, ndf=64, norm_layer=None, use_sigmoid=False, gpu_ids=[], num_D=1):
        super(D_N3Dv1LayersMulti, self).__init__()

        self.gpu_ids = gpu_ids
        self.num_D = num_D
        if num_D == 1:
            layers = self.get_layers(
                input_nc, ndf, norm_layer, use_sigmoid)
            self.model = nn.Sequential(*layers)
        else:
            self.model = ListModule(self, 'model')
            layers = self.get_layers(
                input_nc, ndf, norm_layer, use_sigmoid)
            self.model.append(nn.Sequential(*layers))
            self.down = nn.AvgPool3d(3, stride=2, padding=[
                1, 1, 1], count_include_pad=False)
            for i in range(num_D - 1):
                ndf = int(round(ndf / (2 ** (i + 1))))
                layers = self.get_layers(
                    input_nc, ndf, norm_layer, use_sigmoid)
                self.model.append(nn.Sequential(*layers))

    def get_layers(self, input_nc, ndf=64, norm_layer=nn.BatchNorm3d,
                   use_sigmoid=False):
        kw = 3
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw,
                              stride=1, padding=padw), nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, 5):
            use_stride = 1
            if n in [1, 3]:
                use_stride = 2

            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=use_stride, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = 8
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        return sequence

    def parallel_forward(self, model, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(model, input, self.gpu_ids)
        else:
            return model(input)

    def forward(self, input):
        if self.num_D == 1:
            return self.parallel_forward(self.model, input)
        result = []
        down = input
        for i in range(self.num_D):
            result.append(self.parallel_forward(self.model[i], down))
            if i != self.num_D - 1:
                down = self.parallel_forward(self.down, down)
        return result

class G_Unet_add_all3D(nn.Module):
    """
    Generator network of 3D-BicycleGAN
    """
    def __init__(self, input_nc=15, output_nc=15, nz=8, num_downs=8, ngf=32,
                 norm_layer=None, nl_layer=None, use_dropout=False, gpu_ids=[], upsample='basic'):

        super(G_Unet_add_all3D, self).__init__()
        self.gpu_ids = gpu_ids
        self.nz = nz

        # construct unet blocks
        unet_block = UnetBlock_with_z3D(ngf * 8, ngf * 8, ngf * 8, nz, None, innermost=True,
                                        norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample,
                                        stride=1)
        unet_block = UnetBlock_with_z3D(ngf * 8, ngf * 8, ngf * 8, nz, unet_block,
                                        norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout,
                                        upsample=upsample, stride=1)
        for i in range(num_downs - 6):  # 2 iterations
            if i == 0:
                stride = 2
            elif i == 1:
                stride = 1
            else:
                raise NotImplementedError("Too big, cannot handle!")

            unet_block = UnetBlock_with_z3D(ngf * 8, ngf * 8, ngf * 8, nz, unet_block,
                                            norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout,
                                            upsample=upsample, stride=stride)
        unet_block = UnetBlock_with_z3D(ngf * 4, ngf * 4, ngf * 8, nz, unet_block,
                                        norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample,
                                        stride=2)
        unet_block = UnetBlock_with_z3D(ngf * 2, ngf * 2, ngf * 4, nz, unet_block,
                                        norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample,
                                        stride=1)
        unet_block = UnetBlock_with_z3D(
            ngf, ngf, ngf * 2, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample,
            stride=2)
        unet_block = UnetBlock_with_z3D(input_nc, output_nc, ngf, nz, unet_block,
                                        outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample,
                                        stride=1)
        self.model = unet_block

    def forward(self, x, z):
        return self.model(x, z)


class UnetBlock_with_z3D(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc, nz=0,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='zero',
                 stride=2):
        super(UnetBlock_with_z3D, self).__init__()

        downconv = []
        if padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        self.outermost = outermost
        self.innermost = innermost
        self.nz = nz
        input_nc = input_nc + nz
        downconv += [nn.Conv3d(input_nc, inner_nc,
                               kernel_size=3, stride=stride, padding=p)]

        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nl_layer()

        if outermost:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type, stride=stride)
            down = downconv
            up = [uprelu] + upconv + [nn.Softmax()]
            #original line
            #up = [uprelu] + upconv + [nn.Sigmoid()]
        elif innermost:
            upconv = upsampleLayer(
                inner_nc, outer_nc, upsample=upsample, padding_type=padding_type, stride=stride)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if norm_layer is not None:
                up += [norm_layer(outer_nc)]
        else:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type, stride=stride)
            down = [downrelu] + downconv
            if norm_layer is not None:
                down += [norm_layer(inner_nc)]
            up = [uprelu] + upconv

            if norm_layer is not None:
                up += [norm_layer(outer_nc)]

            if use_dropout:
                up += [nn.Dropout(0.5)]
        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x, z):
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1, 1).expand(
                z.size(0), z.size(1), x.size(2), x.size(3), x.size(4))
            x_and_z = torch.cat([x, z_img], 1)
        else:
            x_and_z = x

        if self.outermost:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return self.up(x2)
        elif self.innermost:
            x1 = self.up(self.down(x_and_z))
            return torch.cat([x1, x], 1)
        else:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return torch.cat([self.up(x2), x], 1)


def upsampleLayer(inplanes, outplanes, upsample='basic', padding_type='zero', stride=2):
    if upsample == 'basic':
        upconv = [nn.ConvTranspose3d(
            inplanes, outplanes, kernel_size=3, stride=stride, padding=1,
            output_padding=1 if stride == 2 else 0)]
    else:
        raise NotImplementedError(
            'upsample layer [%s] not implemented' % upsample)
    return upconv


def fetch_simple_block3d(in_lay, out_lay, nl, norm_layer, stride=1, kw=3, padw=1):
    return [nn.Conv3d(in_lay, out_lay, kernel_size=kw,
                      stride=stride, padding=padw),
            nl(),
            norm_layer(out_lay)]


class E_3DNLayers(nn.Module):
    """
    E network of 3D-BicycleGAN
    """
    def __init__(self, input_nc=15, output_nc=8, ndf=64,
                 norm_layer='instance', nl_layer=None, gpu_ids=[], vaeLike=False):
        super(E_3DNLayers, self).__init__()
        self.gpu_ids = gpu_ids
        self.vaeLike = vaeLike

        nf_mult = 1
        kw, padw = 3, 1

        # Network
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw,
                              stride=1, padding=padw),
                    nl_layer()]
        # Repeats
        sequence.extend(fetch_simple_block3d(ndf, ndf * 2, nl=nl_layer, norm_layer=norm_layer))
        sequence.append(nn.AvgPool3d(2, 2))

        sequence.extend(fetch_simple_block3d(ndf * 2, ndf * 2, nl=nl_layer, norm_layer=norm_layer))
        sequence.append(nn.AvgPool3d(2, 2))

        sequence.extend(fetch_simple_block3d(ndf * 2, ndf * 4, nl=nl_layer, norm_layer=norm_layer))
        sequence.append(nn.AvgPool3d(2, 2))

        sequence.extend(fetch_simple_block3d(ndf * 4, ndf * 4, nl=nl_layer, norm_layer=norm_layer))

        sequence += [nn.AvgPool3d(3)]
        self.conv = nn.Sequential(*sequence)
        self.fc = nn.Sequential(*[nn.Linear(ndf * 4, output_nc)])
        #self.fc = nn.Sequential(*[nn.Linear(ndf * 32, output_nc)])
        if vaeLike:
            self.fcVar = nn.Sequential(*[nn.Linear(ndf * 4, output_nc)])
            #self.fcVar = nn.Sequential(*[nn.Linear(ndf * 32, output_nc)])

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        return output

