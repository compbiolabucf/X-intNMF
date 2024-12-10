""" Training and testing of the model
"""
import os
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score
import torch
import torch.nn.functional as F
from models import init_model_dict, init_optim
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter

import logging
from tqdm import tqdm
from data_adapter import custom___prepare_trte_data


cuda = True if torch.cuda.is_available() else False


def prepare_trte_data(data_folder, view_list, armed_gpu):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_te.csv"), delimiter=','))
    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
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
    
    return data_train_list, data_all_list, idx_dict, labels





def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter, armed_gpu):
    adj_metric = "cosine" # cosine distance
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric, armed_gpu))
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric, armed_gpu))
    
    return adj_train_list, adj_test_list


def train_epoch(data_list, adj_list, label, one_hot_label, sample_weight, model_dict, optim_dict, train_VCDN=True):
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for m in model_dict:
        model_dict[m].train()    
    num_view = len(data_list)
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)].zero_grad()
        ci_loss = 0
        ci = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i]))
        ci_loss = torch.mean(torch.mul(criterion(ci, label),sample_weight))
        ci_loss.backward()
        optim_dict["C{:}".format(i+1)].step()
        loss_dict["C{:}".format(i+1)] = ci_loss.detach().cpu().numpy().item()
    if train_VCDN and num_view >= 2:
        optim_dict["C"].zero_grad()
        c_loss = 0
        ci_list = []
        for i in range(num_view):
            ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i])))
        c = model_dict["C"](ci_list)    
        c_loss = torch.mean(torch.mul(criterion(c, label),sample_weight))
        c_loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()
    
    return loss_dict
    

def test_epoch(data_list, adj_list, te_idx, model_dict):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    for i in range(num_view):
        ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i])))
    if num_view >= 2:
        c = model_dict["C"](ci_list)    
    else:
        c = ci_list[0]
    c = c[te_idx,:]
    prob = F.softmax(c, dim=1).data.cpu().numpy()
    
    return prob


def train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c, 
               num_epoch_pretrain, num_epoch, armed_gpu):
    test_inverval = 50
    num_view = len(view_list)
    dim_hvcdn = pow(num_class,num_view)
    if data_folder == 'ROSMAP':
        adj_parameter = 2
        dim_he_list = [200,200,100]
    if data_folder == 'BRCA':
        adj_parameter = 10
        dim_he_list = [400,400,200]


    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)




    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)

    
    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda(device = armed_gpu)
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda(device = armed_gpu)
        sample_weight_tr = sample_weight_tr.cuda(device = armed_gpu)
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter, armed_gpu)
    dim_list = [x.shape[1] for x in data_tr_list]
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda(device = armed_gpu)
    
    print("\nPretrain GCNs...")
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    for epoch in range(num_epoch_pretrain):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, 
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_VCDN=False)
    print("\nTraining...")
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
    for epoch in range(num_epoch+1):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, 
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)
        if epoch % test_inverval == 0:
            te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
            print("\nTest: Epoch {:d}".format(epoch))
            if num_class == 2:
                print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test F1: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test AUC: {:.3f}".format(roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:,1])))
            else:
                print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test F1 weighted: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
                print("Test F1 macro: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))
            print()








# -----------------------------------------------------------------------------------------------
# Custom train/test
# -----------------------------------------------------------------------------------------------
from typing import List, Dict, Any, Tuple, Union, Literal
import pandas as pd

def custom___eval_one_test(
    omic_layers: List[pd.DataFrame], 
    label_data_series: pd.Series,
    tr_sample_list: List[str],
    te_sample_list: List[str],
    armed_gpu: int,
    target_id: str,


    adj_parameter: int,
    lr_e_pretrain, 
    lr_e, 
    lr_c, 
    num_epoch_pretrain, 
    num_epoch
):
    '''
        Evaluate one test/one split
    '''


    data_tr_list, data_trte_list, trte_idx, labels_trte, dim_he_list, num_class = custom___prepare_trte_data(omic_layers, label_data_series, tr_sample_list, te_sample_list, armed_gpu)


    test_inverval = 50
    num_view = len(omic_layers)
    dim_hvcdn = pow(num_class,num_view)

    logging.fatal(f"dim_hvcdn: {dim_hvcdn}")


    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)

    
    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda(device = armed_gpu)
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda(device = armed_gpu)
        sample_weight_tr = sample_weight_tr.cuda(device = armed_gpu)

    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter, armed_gpu)
    dim_list = [x.shape[1] for x in data_tr_list]
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda(device = armed_gpu)

    
    print("\nPretrain GCNs...")
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    for epoch in tqdm(range(num_epoch_pretrain), desc=f"Pretrain GCNs, GPU {armed_gpu}, target {target_id}"):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, 
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_VCDN=False)
        

    print("\nTraining...")
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
    for epoch in tqdm(range(num_epoch+1), desc=f"Training MOGONET, GPU {armed_gpu}, target {target_id}"):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, 
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)

    Y_test = labels_trte[trte_idx["te"]]
    prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
    pred = prob.argmax(1)
    prob = prob[:,1]


    return {
        'pred': pd.Series(pred).astype(int).tolist(),
        'prob': pd.Series(prob).astype(int).tolist(),
        'ACC': float(accuracy_score(Y_test, pred)),
        'REC': float(recall_score(Y_test, pred)),
        'F1': float(f1_score(Y_test, pred)),
        'MCC': float(matthews_corrcoef(Y_test, pred)),
        'AUROC': float(roc_auc_score(Y_test, prob)),
        'AUPRC': float(average_precision_score(Y_test, prob)),
    }



def custom___evaluate_one_target(
    omic_layers: Union[List[pd.DataFrame], List[Dict[str, Dict]]], 
    testdata: Union[pd.DataFrame, Dict[str, Dict]],
    target_name: str,
    armed_gpu: int,
    target_id: str,

    adj_parameter,
    lr_e_pretrain, 
    lr_e, 
    lr_c, 
    num_epoch_pretrain, 
    num_epoch,

    result_queue = None
):
    # Prepping the data and result
    logging.info(f"Starting evaluation on target {target_name}")

    omic_layers = [pd.DataFrame.from_dict(x, orient='index') for x in omic_layers]
    testdata = pd.DataFrame.from_dict(testdata, orient='index')
    metrics = ['pred', 'prob', 'ACC', 'REC', 'F1', 'MCC', 'AUROC', 'AUPRC']
    results = pd.DataFrame(index = testdata.index, columns = metrics) 

    # Iterate through each test
    for test_id in tqdm(list(testdata.index), desc=f"Evaluating target {testdata} on testdata, GPU {armed_gpu}"):
        # Get sample IDs
        train_sample_ids = testdata.loc[test_id, f'train_sample_ids']
        train_gnd_truth = testdata.loc[test_id, f'train_ground_truth']
        test_sample_ids = testdata.loc[test_id, f'test_sample_ids']
        test_gnd_truth = testdata.loc[test_id, f'test_ground_truth']
        label_data_series = pd.Series(list(train_gnd_truth) + list(test_gnd_truth), index=list(train_sample_ids) + list(test_sample_ids))



        # Train
        result_for_one_test = custom___eval_one_test(
            omic_layers, 
            label_data_series, 
            train_sample_ids, 
            test_sample_ids, 
            armed_gpu,
            target_id,

            adj_parameter, 
            lr_e_pretrain, 
            lr_e, 
            lr_c, 
            num_epoch_pretrain, 
            num_epoch
        )

        logging.info(f"{test_id}, target {target_id} completed")
        # Store the result
        for data_field in result_for_one_test.keys():
            results.at[test_id, data_field] = result_for_one_test[data_field]

    if result_queue is not None:
        return {
            'id': target_id,
            'data': results.to_dict(orient='index')
        }
    else:
        result_queue.put({
            'id': target_id,
            'data': results.to_dict(orient='index')
        })

            
            


