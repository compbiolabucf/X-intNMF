# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành (@bu1th4nh)
# Title: ml.py
# Date: 2024/10/07 23:09:35
# Description: 
# 
# (c) bu1th4nh. All rights reserved
# -----------------------------------------------------------------------------------------------


import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC

from config import *



def test(run_name, train_id, test_id, clinical):
    H = pd.read_parquet(f'{RESULT_PATH}/{run_name}/H.parquet')    
    train_dataset = H.loc[train_id].copy(deep=True).merge(clinical[['ER']], left_index=True, right_index=True)
    test_dataset = H.loc[test_id].copy(deep=True).merge(clinical[['ER']], left_index=True, right_index=True)


    # Split X-Y
    X_train = train_dataset.drop(columns=['ER']).values
    Y_train = train_dataset['ER']
    X_test = test_dataset.drop(columns=['ER']).values
    Y_test = test_dataset['ER']


    cls_results = {}
    for (j, cls_method) in enumerate(METHODS):
        logging.info(f"Processing {cls_method}...")
        # logging.info(f"Feature size: {X_train.shape[1]}, Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

        if(cls_method == "SVM"):                    cls = SVC(probability=True, verbose=False)
        elif(cls_method == "Random Forest"):        cls = RandomForestClassifier(verbose=False)
        elif(cls_method == "Logistic Regression"):  cls = LogisticRegression(max_iter=1000, verbose=False)
        elif(cls_method == "AdaBoost"):             cls = AdaBoostClassifier()
        else: raise ValueError("Invalid classification method")


        cls.fit(X_train, Y_train)
        predicted = cls.predict_proba(X_test)[::,1]

        fpr, tpr, _ = roc_curve(Y_test, predicted)
        auc_val = auc(fpr, tpr)

        cls_results[cls_method] = {
            'fpr' : fpr.tolist(),
            'tpr' : tpr.tolist(),
            'auc' : auc_val,
        }
    return cls_results
        

        
def dataset_prep(
    positive_samples: list,
    negative_samples: list,
):
    pos_train_idx, pos_test_idx = train_test_split(positive_samples, test_size=0.2)#, random_state=69)
    neg_train_idx, neg_test_idx = train_test_split(negative_samples, test_size=0.2)#, random_state=69)

    train_idx = pos_train_idx + neg_train_idx
    test_idx = pos_test_idx + neg_test_idx
    random.shuffle(train_idx)
    random.shuffle(test_idx)

    return train_idx, test_idx



def core_data_init():
    mRNA = pd.read_parquet(f'{ORGL_PATH}/mRNA.parquet')
    miRNA = pd.read_parquet(f'{ORGL_PATH}/miRNA.parquet')
    clinical = pd.read_parquet(f'{ORGL_PATH}/clinical.parquet')
    clinical['ER'] = clinical['ER'].apply(lambda x: 1 if x == 'Positive' else 0)

    common_sample = list(set(mRNA.columns).intersection(miRNA.columns).intersection(clinical.index))
    positive_samples = list((clinical.loc[common_sample, :]).copy(deep=True)[clinical['ER'] == 1].index)
    negative_samples = list((clinical.loc[common_sample, :]).copy(deep=True)[clinical['ER'] == 0].index)

    return positive_samples, negative_samples, clinical
