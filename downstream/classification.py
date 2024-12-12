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
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, precision_score
from sklearn.svm import SVC

from tqdm import tqdm




def evaluate_one_target(H, testdata, methods_list, target):
    # Prepping the data and result
    logging.info("Starting evaluation")

    metrics = ['pred', 'prob', 'ACC', 'PRE', 'REC', 'F1', 'MCC', 'AUROC', 'AUPRC']
    results = {
        method: pd.DataFrame(index = testdata.index, columns = metrics) 
        for method in methods_list
    }

    # Iterate through each test
    for test_id in tqdm(testdata.index, desc=f"Evaluating target {target} on testdata"):
        # Get sample IDs
        train_sample_ids = testdata.loc[test_id, f'train_sample_ids']
        train_gnd_truth = testdata.loc[test_id, f'train_ground_truth']
        test_sample_ids = testdata.loc[test_id, f'test_sample_ids']
        test_gnd_truth = testdata.loc[test_id, f'test_ground_truth']

        # Get train test X/Y
        X_train = H.loc[train_sample_ids].values
        Y_train = np.array(train_gnd_truth)
        X_test = H.loc[test_sample_ids].values
        Y_test = np.array(test_gnd_truth)

        # Evaluate each method
        for cls_method in methods_list:
            if(cls_method == "SVM"):                    cls = SVC(probability=True, verbose=False)
            elif(cls_method == "Random Forest"):        cls = RandomForestClassifier(verbose=False)
            elif(cls_method == "Logistic Regression"):  cls = LogisticRegression(max_iter=1000, verbose=False)
            elif(cls_method == "AdaBoost"):             cls = AdaBoostClassifier()

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
            results[cls_method].at[test_id, 'pred'] = pd.Series(pred).astype(int).tolist()
            results[cls_method].at[test_id, 'prob'] = pd.Series(prob).astype(float).tolist()
            results[cls_method].at[test_id, 'ACC'] = float(ACC)
            results[cls_method].at[test_id, 'PRE'] = float(PRE)
            results[cls_method].at[test_id, 'REC'] = float(REC)
            results[cls_method].at[test_id, 'F1'] = float(F1)
            results[cls_method].at[test_id, 'MCC'] = float(MCC)
            results[cls_method].at[test_id, 'AUROC'] = float(AUROC)
            results[cls_method].at[test_id, 'AUPRC'] = float(AUPRC)

    return results

            
            





