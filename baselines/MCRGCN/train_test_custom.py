# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Main Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: train_test_custom.py
# Date: 2024/11/23 21:15:39
# Description: 
#   - Training logic for MCRGCN, fitted with custom label and dataset
#   - Part of the code is adapted from the original MCRGCN implementation
# 
# (c) 2024 original authors. All rights reserved. 
# Written with dedication in the University of Central Florida, EPCOT and the Magic Kingdom.
# -----------------------------------------------------------------------------------------------




from model.contrast import Contrast
from model.contrastX3 import ContrastX3
from model.heco import HeCo
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, precision_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from torch_geometric.data import Data
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Union, Literal


import logging
import numpy as np
import pandas as pd
import produce_adjacent_matrix
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse as sp
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as fun


# -----------------------------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------------------------
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch .from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch .from_numpy(sparse_mx.data)
    shape = torch .Size(sparse_mx.shape)
    return torch .sparse.FloatTensor(indices, values, shape)


def convert_sampleid_to_numeric(sample_ids, sample_list):
    return [sample_list.index(sample_id) for sample_id in sample_ids]




# -----------------------------------------------------------------------------------------------
# Train
# Data feeds to model has shape (feature, sample) -> need to transpose
# -----------------------------------------------------------------------------------------------
def parallel_train_test_one_target(
    omic_layers: Union[List[pd.DataFrame], List[Dict[str, Dict]]], 
    omic_sims: List[np.ndarray],
    testdata: Union[pd.DataFrame, Dict[str, Dict]],
    armed_gpu: int,
    target_id: str, 
    result_queue: Any = None
):
    omic_layers = [pd.DataFrame.from_dict(x, orient='index').T for x in omic_layers]
    testdata = pd.DataFrame.from_dict(testdata, orient='index')
    results = {}

    # Iterate through each test
    for test_id in tqdm(testdata.index, desc=f"Evaluating label {target_id} on testdata"):
        train_sample_ids = testdata.loc[test_id, f'train_sample_ids']
        train_gnd_truth = testdata.loc[test_id, f'train_ground_truth']
        test_sample_ids = testdata.loc[test_id, f'test_sample_ids']
        test_gnd_truth = testdata.loc[test_id, f'test_ground_truth']
        label_data_series = pd.Series(list(train_gnd_truth) + list(test_gnd_truth), index=list(train_sample_ids) + list(test_sample_ids))

        # Get sample IDs
        train_sample_ids = convert_sampleid_to_numeric(train_sample_ids, list(label_data_series.index))
        test_sample_ids = convert_sampleid_to_numeric(test_sample_ids, list(label_data_series.index))


        # MCRGCN
        result_for_one_test = custom___train_test(
            omic_layers = omic_layers,
            sim_data = omic_sims,
            label_data_series = label_data_series,
            tr_sample_list = list(train_sample_ids),
            te_sample_list = list(test_sample_ids),
            device = armed_gpu
        )


        logging.info(f"{test_id}, target {target_id} completed")
        # Store the result
        results[test_id] = result_for_one_test

    if result_queue is None:
        return {
            'id': target_id,
            'data': results
        }
    else:
        result_queue.put({
            'id': target_id,
            'data': results
        })
    





# -----------------------------------------------------------------------------------------------
# Actual Training
# -----------------------------------------------------------------------------------------------
def custom___train_test(
    omic_layers,
    sim_data,
    label_data_series,
    tr_sample_list,
    te_sample_list,
    device,
):
    n_sample = omic_layers[0].shape[0]
    labels = torch.LongTensor(label_data_series.values).to(device)


    features1 = torch.FloatTensor(omic_layers[0].values).to(device)
    edge_gene_index = torch.LongTensor(sim_data[0]).t().contiguous().to(device)
    cora1 = Data(x=features1, edge_index=edge_gene_index, y=labels)

    if len(omic_layers) == 3:
        features2 = torch.FloatTensor(omic_layers[2].values).to(device)
        edge_methyl_index = torch.LongTensor(sim_data[2]).t().contiguous().to(device)
        cora2 = Data(x=features2, edge_index=edge_methyl_index, y=labels)

    features3 = torch.FloatTensor(omic_layers[1].values).to(device)
    edge_mirna_index = torch.LongTensor(sim_data[1]).t().contiguous().to(device)
    cora3 = Data(x=features3, edge_index=edge_mirna_index, y=labels)

    # logging.fatal(f'tensor_mRNA: {features1.shape}')
    # logging.fatal(f'tensor_miRNA: {features2.shape}')

    train_mask = tr_sample_list
    test_mask = te_sample_list
    
    logging.info(f'[PROCESS {str(device).split(":")[-1]}] train_mask: {train_mask}')
    logging.info(f'[PROCESS {str(device).split(":")[-1]}] test_mask: {test_mask}')

    pos = np.zeros((cora1.y[train_mask].shape[0], cora1.y[train_mask].shape[0]))
    for i in range(cora1.y[train_mask].shape[0]):
        for j in range(cora1.y[train_mask].shape[0]):
            if cora1.y[train_mask][i] == cora1.y[train_mask][j]:
                pos[i][j] = 1
            else:
                pos[i][j] = 0
    logging.info(pos)
    pos = sp.coo_matrix(pos)
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    pos = pos.to_dense().to(device)
    logging.info(pos)


    logging.info(f'[PROCESS {str(device).split(":")[-1]}] features1.shape[1]: {features1.shape[1]}')
    if(len(omic_layers) == 3): logging.info(f'[PROCESS {str(device).split(":")[-1]}] features2.shape[1]: {features2.shape[1]}')
    logging.info(f'[PROCESS {str(device).split(":")[-1]}] features3.shape[1]: {features3.shape[1]}')

    if(len(omic_layers) == 3): 
        model = HeCo(features1.shape[1], features2.shape[1], features3.shape[1], n_sample).to(device)
        criterion=ContrastX3(128, 0.5, 0.5).to(device)
    else:
        model = HeCo(features1.shape[1], None, features3.shape[1], n_sample).to(device)
        criterion=Contrast(128, 0.5, 0.5).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)



    for epoch in range(120):
        model.train()
        optimizer.zero_grad()

        if len(omic_layers) == 3:
            z_ge, z_mp, z_sc = model(cora1,cora2,cora3)
            loss = criterion(z_ge[train_mask], z_mp[train_mask], z_sc[train_mask], pos)
        else:
            z_ge, z_sc = model(cora1,None,cora3)
            loss = criterion(z_ge[train_mask], z_sc[train_mask], pos)
        
        
        # revise - change: loss -> loss.item()
        logging.info(f'[PROCESS {str(device).split(":")[-1]}] epoch: {epoch} loss: {loss.item():.4f}')
        loss.backward()
        optimizer.step()


    model.eval()
    if len(omic_layers) == 3: embeds = model.get_embeds(cora1, cora2, cora3)
    else: embeds = model.get_embeds(cora1, None, cora3)


    embeds_train  = embeds[train_mask]
    embeds_train  = embeds_train.cuda().data.cpu().numpy()
    
    targets_train = cora1.y[train_mask]
    targets_train = targets_train.cuda().data.cpu().numpy()

    cls = MLPClassifier(activation='tanh', max_iter=2000, solver='adam', alpha=0.001, hidden_layer_sizes=(60,30))  # ovr:一对多策略
    cls.fit(embeds_train, targets_train)
    

    embeds_test   = embeds[test_mask]
    embeds_test   = embeds_test.cuda().data.cpu().numpy()

    targets_test  = cora1.y[test_mask]
    targets_test  = targets_test.cuda().data.cpu().numpy()

    pred = cls.predict(embeds_test)
    prob = cls.predict_proba(embeds_test)[::,1]


    return {
        'pred': pd.Series(pred).astype(int).tolist(),
        'prob': pd.Series(prob).astype(float).tolist(),
        'ACC': float(accuracy_score(targets_test, pred)),
        'PRE': float(precision_score(targets_test, pred)),
        'REC': float(recall_score(targets_test, pred)),
        'F1': float(f1_score(targets_test, pred)),
        'MCC': float(matthews_corrcoef(targets_test, pred)),
        'AUROC': float(roc_auc_score(targets_test, prob)),
        'AUPRC': float(average_precision_score(targets_test, prob)),
    }