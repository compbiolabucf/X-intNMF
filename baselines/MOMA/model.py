# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: model.py
# Date: 2024/11/18 16:28:41
# Description: 
# 
# (c) 2024 bu1th4nh. All rights reserved. 
# Written with dedication in the University of Central Florida, EPCOT and the Magic Kingdom.
# -----------------------------------------------------------------------------------------------


import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Union, Literal

import time
import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import  TensorDataset, DataLoader


# Number of Module =128
class mtlAttention128(nn.Module):
    def __init__(self, In_Nodes1,In_Nodes2, Modules):
        super(mtlAttention128, self).__init__()
        self.Modules = Modules
        self.sigmoid = nn.Sigmoid()

        self.task1_FC1_x = nn.Linear(In_Nodes1, Modules,bias=False)
        self.task1_FC1_y = nn.Linear(In_Nodes1, Modules,bias=False)

        self.task2_FC1_x = nn.Linear(In_Nodes2, Modules,bias=False)
        self.task2_FC1_y = nn.Linear(In_Nodes2, Modules,bias=False)
            
        self.softmax  = nn.Softmax(dim=-1)
        
        self.task1_FC2 =nn.Sequential(nn.Linear(Modules*2, 128),nn.ReLU())
        self.task2_FC2 = nn.Sequential(nn.Linear(Modules*2, 128),nn.ReLU())

        self.task1_FC3 =nn.Sequential(nn.Linear(128, 64),nn.ReLU())
        self.task2_FC3 = nn.Sequential(nn.Linear(128, 64),nn.ReLU())
        
        self.task1_FC4 =nn.Sequential(nn.Linear(64, 32),nn.ReLU())
        self.task2_FC4 = nn.Sequential(nn.Linear(64, 32),nn.ReLU())
        
        self.task1_FC5 =nn.Sequential(nn.Linear(32, 16),nn.ReLU())
        self.task2_FC5 = nn.Sequential(nn.Linear(32, 16),nn.ReLU())
        
        self.task1_FC6 = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())
        self.task2_FC6 = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())

    def forward_one(self,xg,xm):
        xg_x = self.task1_FC1_x(xg)
        xm_x = self.task2_FC1_x(xm)
        xg_y = self.task1_FC1_y(xg)         
        xm_y =self.task2_FC1_y(xm)

        xg = torch.cat([xg_x.reshape(-1,1,self.Modules),xg_y.reshape(-1,1,self.Modules)], dim=1)
        xm = torch.cat([xm_x.reshape(-1,1,self.Modules),xm_y.reshape(-1,1,self.Modules)], dim=1)
        
        norm  = torch.norm(xg, dim=1, keepdim=True)
        xg = xg.div(norm)
        
        norm  = torch.norm(xm, dim=1, keepdim=True)
        xm = xm.div(norm)
        
        energy =  torch.bmm(xg.reshape(-1,2,self.Modules).permute(0,2,1) ,xm.reshape(-1,2,self.Modules))
        attention1 = self.softmax(energy.permute(0,2,1)).permute(0,2,1) 
        attention2 = self.softmax(energy).permute(0,2,1)
        
        xg_value = torch.bmm(xg,attention1) 
        xm_value = torch.bmm(xm,attention2)

        xg = xg_value.view(-1,self.Modules*2)
        xm =xm_value.view(-1,self.Modules*2)
        
        xg = self.task1_FC2(xg)
        xm = self.task2_FC2(xm) 
        xg = self.task1_FC3(xg)
        xm = self.task2_FC3(xm)
        xg = self.task1_FC4(xg)
        xm = self.task2_FC4(xm)
        xg = self.task1_FC5(xg)
        xm = self.task2_FC5(xm)
        xg = self.task1_FC6(xg)
        xm = self.task2_FC6(xm)
        
        return xg,xm
    

# Number of Module = 64
class mtlAttention64(nn.Module):
    def __init__(self, In_Nodes1,In_Nodes2, Modules):
        super(mtlAttention64, self).__init__()
        self.Modules = Modules
        self.sigmoid = nn.Sigmoid()

        self.task1_FC1_x = nn.Linear(In_Nodes1, Modules,bias=False)
        self.task1_FC1_y = nn.Linear(In_Nodes1, Modules,bias=False)

        self.task2_FC1_x = nn.Linear(In_Nodes2, Modules,bias=False)
        self.task2_FC1_y = nn.Linear(In_Nodes2, Modules,bias=False)
            
        self.softmax  = nn.Softmax(dim=-1)
        
        self.task1_FC2 =nn.Sequential(nn.Linear(Modules*2, 64),nn.ReLU())
        self.task2_FC2 = nn.Sequential(nn.Linear(Modules*2, 64),nn.ReLU())

        self.task1_FC3 =nn.Sequential(nn.Linear(64, 32),nn.ReLU())
        self.task2_FC3 = nn.Sequential(nn.Linear(64, 32),nn.ReLU())
        
        self.task1_FC4 =nn.Sequential(nn.Linear(32, 16),nn.ReLU())
        self.task2_FC4 = nn.Sequential(nn.Linear(32, 16),nn.ReLU())
        
        self.task1_FC5 = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())
        self.task2_FC5 = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())

    def forward_one(self,xg,xm):
        xg_x = self.task1_FC1_x(xg)
        xm_x = self.task2_FC1_x(xm)
        xg_y = self.task1_FC1_y(xg)         
        xm_y =self.task2_FC1_y(xm)

        xg = torch.cat([xg_x.reshape(-1,1,self.Modules),xg_y.reshape(-1,1,self.Modules)], dim=1)
        xm = torch.cat([xm_x.reshape(-1,1,self.Modules),xm_y.reshape(-1,1,self.Modules)], dim=1)
        
        norm  = torch.norm(xg, dim=1, keepdim=True)
        xg = xg.div(norm)
        
        norm  = torch.norm(xm, dim=1, keepdim=True)
        xm = xm.div(norm)
        
        energy =  torch.bmm(xg.reshape(-1,2,self.Modules).permute(0,2,1) ,xm.reshape(-1,2,self.Modules))
        attention1 = self.softmax(energy.permute(0,2,1)).permute(0,2,1) 
        attention2 = self.softmax(energy).permute(0,2,1)
        
        xg_value = torch.bmm(xg,attention1) 
        xm_value = torch.bmm(xm,attention2)

        xg = xg_value.view(-1,self.Modules*2)
        xm =xm_value.view(-1,self.Modules*2)
        
        xg = self.task1_FC2(xg)
        xm = self.task2_FC2(xm) 
        xg = self.task1_FC3(xg)
        xm = self.task2_FC3(xm)
        xg = self.task1_FC4(xg)
        xm = self.task2_FC4(xm)
        xg = self.task1_FC5(xg)
        xm = self.task2_FC5(xm)
        
        return xg,xm


# Number of Module = 32
class mtlAttention32(nn.Module):
    def __init__(self, In_Nodes1,In_Nodes2, Modules):
        super(mtlAttention32, self).__init__()
        self.Modules = Modules
        self.sigmoid = nn.Sigmoid()

        self.task1_FC1_x = nn.Linear(In_Nodes1, Modules,bias=False)
        self.task1_FC1_y = nn.Linear(In_Nodes1, Modules,bias=False)

        self.task2_FC1_x = nn.Linear(In_Nodes2, Modules,bias=False)
        self.task2_FC1_y = nn.Linear(In_Nodes2, Modules,bias=False)
            
        self.softmax  = nn.Softmax(dim=-1)
        
        self.task1_FC2 =nn.Sequential(nn.Linear(Modules*2, 32),nn.ReLU())
        self.task2_FC2 = nn.Sequential(nn.Linear(Modules*2, 32),nn.ReLU())

        self.task1_FC3 =nn.Sequential(nn.Linear(32, 16),nn.ReLU())
        self.task2_FC3 = nn.Sequential(nn.Linear(32, 16),nn.ReLU())
        
        self.task1_FC4 = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())
        self.task2_FC4 = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())

    def forward_one(self,xg,xm):
        xg_x = self.task1_FC1_x(xg)
        xm_x = self.task2_FC1_x(xm)
        xg_y = self.task1_FC1_y(xg)         
        xm_y =self.task2_FC1_y(xm)

        xg = torch.cat([xg_x.reshape(-1,1,self.Modules),xg_y.reshape(-1,1,self.Modules)], dim=1)
        xm = torch.cat([xm_x.reshape(-1,1,self.Modules),xm_y.reshape(-1,1,self.Modules)], dim=1)
        
        norm  = torch.norm(xg, dim=1, keepdim=True)
        xg = xg.div(norm)
        
        norm  = torch.norm(xm, dim=1, keepdim=True)
        xm = xm.div(norm)
        
        energy =  torch.bmm(xg.reshape(-1,2,self.Modules).permute(0,2,1) ,xm.reshape(-1,2,self.Modules))
        attention1 = self.softmax(energy.permute(0,2,1)).permute(0,2,1) 
        attention2 = self.softmax(energy).permute(0,2,1)
        
        xg_value = torch.bmm(xg,attention1) 
        xm_value = torch.bmm(xm,attention2)

        xg = xg_value.view(-1,self.Modules*2)
        xm =xm_value.view(-1,self.Modules*2)
        
        xg = self.task1_FC2(xg)
        xm = self.task2_FC2(xm) 
        xg = self.task1_FC3(xg)
        xm = self.task2_FC3(xm)
        xg = self.task1_FC4(xg)
        xm = self.task2_FC4(xm)
        
        return xg,xm


# Number of Module = 16
class mtlAttention16(nn.Module):
    def __init__(self, In_Nodes1,In_Nodes2, Modules):
        super(mtlAttention16, self).__init__()
        self.Modules = Modules
        self.sigmoid = nn.Sigmoid()

        self.task1_FC1_x = nn.Linear(In_Nodes1, Modules,bias=False)
        self.task1_FC1_y = nn.Linear(In_Nodes1, Modules,bias=False)

        self.task2_FC1_x = nn.Linear(In_Nodes2, Modules,bias=False)
        self.task2_FC1_y = nn.Linear(In_Nodes2, Modules,bias=False)
            
        self.softmax  = nn.Softmax(dim=-1)
        
        self.task1_FC2 =nn.Sequential(nn.Linear(Modules*2, 16),nn.ReLU())
        self.task2_FC2 = nn.Sequential(nn.Linear(Modules*2, 16),nn.ReLU())
        
        self.task1_FC3 = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())
        self.task2_FC3 = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())

    def forward_one(self,xg,xm):
        xg_x = self.task1_FC1_x(xg)
        xm_x = self.task2_FC1_x(xm)
        xg_y = self.task1_FC1_y(xg)         
        xm_y =self.task2_FC1_y(xm)

        xg = torch.cat([xg_x.reshape(-1,1,self.Modules),xg_y.reshape(-1,1,self.Modules)], dim=1)
        xm = torch.cat([xm_x.reshape(-1,1,self.Modules),xm_y.reshape(-1,1,self.Modules)], dim=1)
        
        norm  = torch.norm(xg, dim=1, keepdim=True)
        xg = xg.div(norm)
        
        norm  = torch.norm(xm, dim=1, keepdim=True)
        xm = xm.div(norm)
        
        energy =  torch.bmm(xg.reshape(-1,2,self.Modules).permute(0,2,1) ,xm.reshape(-1,2,self.Modules))
        attention1 = self.softmax(energy.permute(0,2,1)).permute(0,2,1) 
        attention2 = self.softmax(energy).permute(0,2,1)
        
        xg_value = torch.bmm(xg,attention1) 
        xm_value = torch.bmm(xm,attention2)

        xg = xg_value.view(-1,self.Modules*2)
        xm =xm_value.view(-1,self.Modules*2)
        
        xg = self.task1_FC2(xg)
        xm = self.task2_FC2(xm) 
        xg = self.task1_FC3(xg)
        xm = self.task2_FC3(xm)
        
        return xg,xm


# Early Stopping
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.0001, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >self.patience:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss