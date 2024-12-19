# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: classification.py
# Date: 2024/12/07 11:20:53
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




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, precision_score
from sklearn.svm import SVC

from tqdm import tqdm



def evaluate_one_test(H, train_sample_ids, train_gnd_truth, test_sample_ids, test_gnd_truth, classifier):
    if(classifier == "SVM"):                    cls = SVC(probability=True, verbose=False)
    elif(classifier == "Random Forest"):        cls = RandomForestClassifier(verbose=False)
    elif(classifier == "Logistic Regression"):  cls = LogisticRegression(max_iter=1000, verbose=False)
    elif(classifier == "AdaBoost"):             cls = AdaBoostClassifier()

    # Get train test X/Y
    X_train = H.loc[train_sample_ids].values
    Y_train = np.array(train_gnd_truth)
    X_test = H.loc[test_sample_ids].values
    Y_test = np.array(test_gnd_truth)

    # Fit & predict the model
    cls.fit(X_train, Y_train)
    pred = cls.predict(X_test)
    prob = cls.predict_proba(X_test)[::,1]

    # Metrics
    ACC = accuracy_score(Y_test, pred)
    PRE = precision_score(Y_test, pred)
    REC = recall_score(Y_test, pred)
    F1 = f1_score(Y_test, pred)
    MCC = matthews_corrcoef(Y_test, pred)
    AUROC = roc_auc_score(Y_test, prob)
    AUPRC = average_precision_score(Y_test, prob)

    # Store the result
    return {
        'pred': pd.Series(pred).astype(int).tolist(),
        'prob': pd.Series(prob).astype(float).tolist(),
        'ACC': float(ACC),
        'PRE': float(PRE),
        'REC': float(REC),
        'F1': float(F1),
        'MCC': float(MCC),
        'AUROC': float(AUROC),
        'AUPRC': float(AUPRC),
    }



def evaluate_one_target(H, testdata, methods_list, target):
    # Prepping the data and result
    logging.info("Starting evaluation")


    results = {
        method: {} for method in methods_list
    }

    # Iterate through each test
    for test_id in tqdm(testdata.index, desc=f"Evaluating target {target} on testdata"):
        # Get sample IDs
        train_sample_ids = testdata.loc[test_id, f'train_sample_ids']
        train_gnd_truth = testdata.loc[test_id, f'train_ground_truth']
        test_sample_ids = testdata.loc[test_id, f'test_sample_ids']
        test_gnd_truth = testdata.loc[test_id, f'test_ground_truth']

        # Evaluate each method
        for classifier in methods_list:
            results[classifier][test_id] = evaluate_one_test(H, train_sample_ids, train_gnd_truth, test_sample_ids, test_gnd_truth, classifier)

    return results

            
            





def hparams_evaluate(H, train_sample_ids, train_gnd_truth, classifier):
    # Prepping the data and result
    logging.info("Starting evaluation")

    if(classifier == "SVM"):                    estimator = SVC(probability=True, verbose=False)
    elif(classifier == "Random Forest"):        estimator = RandomForestClassifier(verbose=False)
    elif(classifier == "Logistic Regression"):  estimator = LogisticRegression(max_iter=1000, verbose=False)
    elif(classifier == "AdaBoost"):             estimator = AdaBoostClassifier()

    Ariel = cross_validate(
        estimator=estimator,
        X=H.loc[train_sample_ids].values,
        y=train_gnd_truth,
        scoring={
            'ACC': 'accuracy',
            'PRE': 'precision',
            'REC': 'recall',
            'F1': 'f1',
            'MCC': 'matthews_corrcoef',
            'AUROC': 'roc_auc',
            'AUPRC': 'average_precision',
        }
    )
    for key in Ariel.keys(): Ariel[key] = list(Ariel[key]) # Sanitize the numpy array to list
    return {
        'details': Ariel,
        'AUROC': float(np.mean(Ariel['test_AUROC'])),
        'MCC': float(np.mean(Ariel['test_MCC'])),
    }
    
    
          

class HParamsParallelWrapper:
    def __init__(
        self,
        run_cfg_data,
        tar_data,
        dataset_id,
    ):
        self.run_cfg_data = run_cfg_data
        self.tar_data = tar_data
        self.dataset_id = dataset_id


    def __call__(self, target_id, test_id, config, classifier):
        H = self.run_cfg_data[config].copy(deep=True)
        train_sample_ids = list(self.tar_data[target_id].loc[test_id, f'train_sample_ids'])
        train_gnd_truth = list(self.tar_data[target_id].loc[test_id, f'train_ground_truth'])
        data_pack = hparams_evaluate(H, train_sample_ids, train_gnd_truth, classifier, config)

        data_pack.update({
            'dataset': self.dataset_id,
            'target_id': target_id,
            'test_id': test_id,
            'config': config,
            'classifier': classifier,
        })


        del H
        return data_pack

    def eval_nonparallel(self, input):
        target_id = input['target_id']
        test_id = input['test_id']
        config = input['config']
        classifier = input['classifier']
        H = self.run_cfg_data[config].copy(deep=True)
        train_sample_ids = list(self.tar_data[target_id].loc[test_id, f'train_sample_ids'])
        train_gnd_truth = list(self.tar_data[target_id].loc[test_id, f'train_ground_truth'])
        data_pack = hparams_evaluate(H, train_sample_ids, train_gnd_truth, classifier)

        data_pack.update({
            'dataset': self.dataset_id,
            'target_id': target_id,
            'test_id': test_id,
            'config': config,
            'classifier': classifier,
        })


        del H
        return data_pack