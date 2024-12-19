# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: data_adapter.py
# Date: 2024/11/17 13:34:13
# Description: Adapter for data from our format to the format of the original MOGONET code.
# 
# (c) 2024 bu1th4nh. All rights reserved. 
# Written with dedication in the University of Central Florida, EPCOT and the Magic Kingdom.
# -----------------------------------------------------------------------------------------------

import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Union, Literal


# -----------------------------------------------------------------------------------------------
# Train/Test Preparation
# -----------------------------------------------------------------------------------------------
def custom___prepare_trte_data(
    omic_layers: List[pd.DataFrame], 
    label_data_series: pd.Series,
    tr_sample_list: List[str],
    te_sample_list: List[str],
    armed_gpu: int,
) -> Tuple[
    List[torch.Tensor], 
    List[torch.Tensor], 
    Dict[str, List[int]], 
    np.ndarray,
    list,
    int,
]:
    # Set up num class and dim_he_list
    num_class = len(label_data_series.unique())
    dim_he_list = [layer.shape[0] for layer in omic_layers]


    # Retrieve train/test labels
    num_view = len(omic_layers)
    labels_tr = label_data_series.loc[tr_sample_list].values.astype(int)
    labels_te = label_data_series.loc[te_sample_list].values.astype(int)


    # Retrieve train/test data
    data_tr_list = []
    data_te_list = []
    for i in range(num_view):
        data_tr_list.append(omic_layers[i].T.loc[tr_sample_list].values)
        data_te_list.append(omic_layers[i].T.loc[te_sample_list].values)
    

    # Concatenate train/test data and create tensor
    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    
    
    
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if torch.cuda.is_available():
            data_tensor_list[i] = data_tensor_list[i].cuda(device = armed_gpu)
    
    
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))


    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_tensor_list[i][idx_dict["te"]].clone()),0))
    labels = np.concatenate((labels_tr, labels_te))
    
    return data_train_list, data_all_list, idx_dict, labels, dim_he_list, num_class


