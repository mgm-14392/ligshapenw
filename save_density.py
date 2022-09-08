#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2021 Institut Pasteur                                       #
#                 				                            #
#                                                                           #
#  Redistribution and use in source and binary forms, with or without       #
#  modification, are permitted provided that the following conditions       #
#  are met:                                                                 #
#                                                                           #
#  1. Redistributions of source code must retain the above copyright        #
#  notice, this list of conditions and the following disclaimer.            #
#  2. Redistributions in binary form must reproduce the above copyright     #
#  notice, this list of conditions and the following disclaimer in the      #
#  documentation and/or other materials provided with the distribution.     #
#  3. Neither the name of the copyright holder nor the names of its         #
#  contributors may be used to endorse or promote products derived from     #
#  this software without specific prior written permission.                 #
#                                                                           #
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      #
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        #
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    #
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     #
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   #
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         #
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    #
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    #
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      #
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    #
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     #
#                                                                           #
#  This program is free software: you can redistribute it and/or modify     #
#                                                                           #
#############################################################################

import mrcfile
import numpy as np
import torch


def save_density(density, outfilename, spacing, origin, padding):
    """
    Save the density file as mrc for the given atomname
    """
    density = density.astype('float32')
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(density.T)
        mrc.voxel_size = spacing
        mrc.header['origin']['x'] = origin[0] - padding + .5 * spacing
        mrc.header['origin']['y'] = origin[1] - padding + .5 * spacing
        mrc.header['origin']['z'] = origin[2] - padding + .5 * spacing
        mrc.update_header_from_data()
        mrc.update_header_stats()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-t', '--tensor')
    parser.add_argument('-o', '--origin', type=float, nargs='+')  # 10. 20. 30.
    parser.add_argument('-s', '--spacing', type=float)  # 1.
    parser.add_argument('--out')  # 10. 20. 30.
    args = parser.parse_args()

    data1 = torch.load(args.tensor, map_location=torch.device('cpu'))
    batch_size = data1.shape[0]
    data2 = torch.chunk(data1, batch_size, dim=0)
    data = [torch.squeeze(i) for i in data2]
    print(data[0].shape)

    # loop batch
    for i in range(len(data)):
        # loop channel
        for j in range(len(data[0])):
            np_matrix = data[i][j].detach().numpy()
            out = "%s_%d_channel%d.mrc"%(args.out, i, j)
            save_density(np_matrix, out, args.spacing, args.origin, 0)
