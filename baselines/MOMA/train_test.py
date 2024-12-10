# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: train_test.py
# Date: 2024/11/18 17:02:38
# Description: 
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
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import  TensorDataset, DataLoader


from model import mtlAttention32, EarlyStopping


def custom___train_test(
    omic_layers,
    label_data_series,
    tr_sample_list,
    te_sample_list,
    device,
):
    earlyStoppingPatience = 50
    learningRate = 0.000005
    weightDecay = 0.001
    num_epochs = 50000 

    y_train = label_data_series.loc[tr_sample_list].values.astype(int)
    y_test = label_data_series.loc[te_sample_list].values.astype(int)

    Xg = torch.tensor(omic_layers[0].T.loc[tr_sample_list].values, dtype=torch.float32).cuda(device=device)
    Xg_test = torch.tensor(omic_layers[0].T.loc[te_sample_list].values, dtype=torch.float32).cuda(device=device)

    Xm = torch.tensor(omic_layers[1].T.loc[tr_sample_list].values, dtype=torch.float32).cuda(device=device)
    Xm_test = torch.tensor(omic_layers[1].T.loc[te_sample_list].values, dtype=torch.float32).cuda(device=device)

    y = torch.tensor(y_train, dtype=torch.float32).cuda(device=device)

    ds = TensorDataset(Xg, Xm,y)
    loader  = DataLoader(ds, batch_size=y_train.shape[0],shuffle=True)

    Xg_test = torch.tensor(Xg_test, dtype=torch.float32).cuda(device=device)
    Xm_test = torch.tensor(Xm_test, dtype=torch.float32).cuda(device=device)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    In_Nodes1 = omic_layers[0].shape[0] 
    In_Nodes2 = omic_layers[1].shape[0]

    # mtlAttention(In_Nodes1,In_Nodes2, # of module)
    net = mtlAttention32(In_Nodes1,In_Nodes2,32)
    net = net.to(device)
    early_stopping = EarlyStopping(patience=earlyStoppingPatience, verbose=False)
    optimizer = optim.Adam(net.parameters(), lr=learningRate, weight_decay=weightDecay)
    loss_fn = nn.BCELoss()

    for epoch in tqdm(range(num_epochs)):
        running_loss1 = 0.0
        running_loss2 = 0.0
        for i, data in enumerate(loader, 0):
            xg,xm, y = data
            output1,output2 = net.forward_one(xg,xm)
            output1  = output1.squeeze()
            output2  = output2.squeeze()
            net.train()
            optimizer.zero_grad()
            loss = loss_fn(output1, y) + loss_fn(output2, y) 
            loss.backward(retain_graph=True)
            optimizer.step()
            running_loss1 += loss_fn(output1,y.view(-1)).item()
            running_loss2 += loss_fn(output2,y.view(-1)).item()


        early_stopping(running_loss1+running_loss2, net)
        if early_stopping.early_stop:
            logging.info(f"Early stopping at epoch {epoch}")
            logging.info("--------------------------------------------------------------------------------------------------")
            break


                
    ### Test
    test1,test2 = net.forward_one(Xg_test.clone().detach(),Xm_test.clone().detach())
    test1 = test1.cpu().detach().numpy()
    test2 = test2.cpu().detach().numpy()


    prob = (test1 + test2)/2
    pred = np.where(prob > 0.5, 1, 0) # Change the method from avg AUC to avg prob then predict
    
    return {
        'pred': pd.Series(pred).astype(int).tolist(),
        'prob': pd.Series(prob).astype(float).tolist(),
        'ACC': float(accuracy_score(y_test, pred)),
        'REC': float(recall_score(y_test, pred)),
        'F1': float(f1_score(y_test, pred)),
        'MCC': float(matthews_corrcoef(y_test, pred)),
        'AUROC': float(roc_auc_score(y_test, prob)),
        'AUPRC': float(average_precision_score(y_test, prob)),
    }









def parallel_train_test_one_target(omic_layers: Union[List[pd.DataFrame], List[Dict[str, Dict]]], 
    testdata: Union[pd.DataFrame, Dict[str, Dict]],
    target_name: str,
    armed_gpu: int,
    target_id: str, 
    result_queue: Any = None
):
    omic_layers = [pd.DataFrame.from_dict(x, orient='index') for x in omic_layers]
    testdata = pd.DataFrame.from_dict(testdata, orient='index')
    metrics = ['pred', 'prob', 'ACC', 'REC', 'F1', 'MCC', 'AUROC', 'AUPRC']
    results = pd.DataFrame(index = testdata.index, columns = metrics) 

    # Iterate through each test
    for test_id in tqdm(testdata.index, desc=f"Evaluating label {target_name} on testdata"):
        # Get sample IDs
        train_sample_ids = testdata.loc[test_id, f'train_sample_ids']
        train_gnd_truth = testdata.loc[test_id, f'train_ground_truth']
        test_sample_ids = testdata.loc[test_id, f'test_sample_ids']
        test_gnd_truth = testdata.loc[test_id, f'test_ground_truth']
        label_data_series = pd.Series(list(train_gnd_truth) + list(test_gnd_truth), index=list(train_sample_ids) + list(test_sample_ids))


        # MOMA
        result_for_one_test = custom___train_test(
            omic_layers = omic_layers,
            label_data_series = label_data_series,
            tr_sample_list = list(train_sample_ids),
            te_sample_list = list(test_sample_ids),
            device = armed_gpu
        )


        logging.info(f"{test_id}, target {target_id} completed")
        # Store the result
        for data_field in result_for_one_test.keys():
            results.at[test_id, data_field] = result_for_one_test[data_field]

    if result_queue is None:
        return {
            'id': target_id,
            'data': results.to_dict(orient='index')
        }
    else:
        result_queue.put({
            'id': target_id,
            'data': results.to_dict(orient='index')
        })
    



