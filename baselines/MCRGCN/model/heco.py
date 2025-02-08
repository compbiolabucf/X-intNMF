import torch.nn as nn
import torch
from model.Methy_model import MethyGCN
from model.Mirna_model import MirnaGCN
from model.Gene_model import GeneGCN
from model.contrast import Contrast
import torch.nn.functional as fun

import numpy as np
from torch_geometric.data import Data


class HeCo(nn.Module):
    methyl_activated = False

    def __init__(self, num_feature1, num_feature2, num_feature3, n_sample):
        super(HeCo, self).__init__()
        self.ge = GeneGCN(num_feature1, n_sample)
        self.sc = MirnaGCN(num_feature3, n_sample)

        if num_feature2 is not None:
            self.mp = MethyGCN(num_feature2, n_sample)
            self.methyl_activated = True

        self.LP3 = torch.nn.Linear(128, 256)
        self.LP4 = torch.nn.Linear(256, 128)
        

    def forward(self, data1, data2, data3):  # p a s
        z_ge = self.ge(data1)
        z_ge = self.LP3(z_ge)
        z_ge = fun.silu(z_ge)
        z_ge = self.LP4(z_ge)


        if self.methyl_activated:
            z_mp = self.mp(data2)
            z_mp = self.LP3(z_mp)
            z_mp = fun.silu(z_mp)
            z_mp = self.LP4(z_mp)


        z_sc = self.sc(data3)
        z_sc = self.LP3(z_sc)
        z_sc = fun.silu(z_sc)
        z_sc = self.LP4(z_sc)


        return z_ge, z_sc

    def get_embeds(self, data1, data2, data3):
        z_ge = self.ge(data1)
        z_ge = self.LP3(z_ge)
        z_ge = fun.silu(z_ge)
        z_ge = self.LP4(z_ge)


        if self.methyl_activated:
            z_mp = self.mp(data2)
            z_mp = self.LP3(z_mp)
            z_mp = fun.silu(z_mp)
            z_mp = self.LP4(z_mp)



        z_sc = self.sc(data3)
        z_sc = self.LP3(z_sc)
        z_sc = fun.silu(z_sc)
        z_sc = self.LP4(z_sc)


        if self.methyl_activated: z = z_ge + z_sc + z_mp
        else: z = z_ge + z_sc
        
        return z_ge.detach()

    # z_ge.detach()
