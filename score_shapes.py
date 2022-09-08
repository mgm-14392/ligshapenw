import torch
import sys
from os import listdir
from os.path import isfile, join
import sys
import baseoptions
import csv

def cross_entropy(real_tensor, fake_tensor, eps=1e-5):
    CE = -1*((real_tensor * torch.log(fake_tensor + eps)).sum(dim=1).mean())
    return CE

def channel_pearson_corr(output_tensor,target_tensor):
    # mean by channel and reshape to do substraction
    voutput_tensor = output_tensor - output_tensor.mean(dim = [2,3,4]) \
        .view(output_tensor.shape[0], output_tensor.shape[1], 1, 1, 1)
    vtarget_tensor = target_tensor - target_tensor.mean(dim = [2,3,4]) \
        .view(target_tensor.shape[0], target_tensor.shape[1], 1, 1, 1)
    # pearson correlation between each channels but for all elements in the batch. cost.shape would be = [14]
    pc = torch.sum((voutput_tensor * vtarget_tensor),
                   dim = [0,2,3,4]) / (torch.sqrt(torch.sum((voutput_tensor ** 2),
                                                            dim = [0,2,3,4])) * torch.sqrt(torch.sum((vtarget_tensor ** 2),
                                                                                                     dim = [0,2,3,4])))
    # pearson correlation between each channels for each element in batch change dim to [2,3,4]. cost.shape would be = [batch_size,14]
    return pc

def pearson_Corr(output_tensor,target_tensor):
    voutput_tensor = output_tensor - torch.mean(output_tensor)
    vtarget_tensor = target_tensor - torch.mean(target_tensor)
    cost = torch.sum(voutput_tensor * vtarget_tensor) / (torch.sqrt(torch.sum(voutput_tensor ** 2)) * torch.sqrt(torch.sum(vtarget_tensor ** 2)))
    return cost


if __name__ == '__main__':
    args = baseoptions.BaseOptions().create_parser()

    mypath = args['dir_ligshape']
    allfiles = [ join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

    onlyfiles = [val for val in allfiles if val.endswith(".pt")]

    fce = open("CE_gen_ligshape.csv",'w')
    fpc = open("PC_gen_ligshape.csv",'w')
    fcpc = open("cPC_gen_ligshape.csv",'w')


    for _file1 in onlyfiles:
        shapev1 = torch.load(_file1)
        file_name1 = _file1.rpartition("/")[-1].rpartition(".")[0]

        for _file2 in onlyfiles:
            shapev2 = torch.load(_file2)
            file_name2 = _file2.rpartition("/")[-1].rpartition(".")[0]

            key = "%s_%s"%(file_name1,file_name2)

            CE = cross_entropy(shapev1, shapev2)
            cPC = channel_pearson_corr(shapev1, shapev2)
            PC = pearson_Corr(shapev1, shapev2)

            fce.write("%s CE %.2f n/" % (key, CE.item()))
            fpc.write("%s OPC %.2f n/"%(key, PC.item()))
            fcpc.write("%s CPC %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f n/"%(key, cPC[0].item(),cPC[1].item(),cPC[2].item(),cPC[3].item(),cPC[4].item(),cPC[5].item(),cPC[6].item(),cPC[7].item(),cPC[8].item(),cPC[9].item(),cPC[10].item(),cPC[11].item(),cPC[12].item(),cPC[13].item(),cPC[14].item()))

    fce.close()
    fpc.close()
    fcpc.close()