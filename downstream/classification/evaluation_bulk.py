# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành (@bu1th4nh)
# Title: evaluation_bulk.py
# Date: 2024/10/19 20:13:59
# Description: 
# 
# (c) bu1th4nh. All rights reserved
# -----------------------------------------------------------------------------------------------


import mlflow
import logging
import cupy as cp
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC

from tqdm import tqdm



def evaluate(H, label_data, testdata, methods_list):
    # Prepping the data and result
    logging.info("Starting evaluation")
    auc_columns = []
    for label in label_data.columns:
        for method in methods_list:
            auc_columns.append(f"{label}_{method}_AUC")
            auc_columns.append(f"{label}_{method}_TPR")
            auc_columns.append(f"{label}_{method}_FPR")
    AUC_result = pd.DataFrame(
        index = testdata.index,
        columns = auc_columns
    )

    # Assume label is 0-1
    for label in label_data.columns:
        # Iterate through each test
        for test_id in tqdm(testdata.index, desc=f"Evaluating label {label} on testdata"):
            # Get sample IDs
            train_sample_ids = testdata.loc[test_id, f'{label}_train']
            test_sample_ids = testdata.loc[test_id, f'{label}_test']

            # Get train test X/Y
            X_train = H.loc[train_sample_ids].values
            Y_train = label_data.loc[train_sample_ids, label]
            X_test = H.loc[test_sample_ids].values
            Y_test = label_data.loc[test_sample_ids, label]

            # Evaluate each method
            for cls_method in methods_list:
                if(cls_method == "SVM"):                    cls = SVC(probability=True, verbose=False)
                elif(cls_method == "Random Forest"):        cls = RandomForestClassifier(verbose=False)
                elif(cls_method == "Logistic Regression"):  cls = LogisticRegression(max_iter=1000, verbose=False)
                elif(cls_method == "AdaBoost"):             cls = AdaBoostClassifier()

                # Fit the model
                cls.fit(X_train, Y_train)
                predicted = cls.predict_proba(X_test)[::,1]
                fpr, tpr, _ = roc_curve(Y_test, predicted)
                auc_val = auc(fpr, tpr)

                # Store the result
                AUC_result.at[test_id, f"{label}_{cls_method}_AUC"] = auc_val
                AUC_result.at[test_id, f"{label}_{cls_method}_TPR"] = tpr
                AUC_result.at[test_id, f"{label}_{cls_method}_FPR"] = fpr


    # Calculate, log statistics
    logging.info("Eval completed. Calculating statistics")
    for label in label_data.columns:
        for cls_method in methods_list:
            auc_values = AUC_result[f"{label}_{cls_method}_AUC"].values
            avg_auc = np.mean(auc_values)
            std_auc = np.std(auc_values)
            max_auc = np.max(auc_values)
            min_auc = np.min(auc_values)
            med_auc = np.median(auc_values)
            
            
            # Logging
            logging.info(f"{label} - {method} - Mean AUC: {avg_auc:.05f}")
            logging.info(f"{label} - {method} - Median AUC: {med_auc:.05f}")
            logging.info(f"{label} - {method} - Std AUC: {std_auc:.05f}")
            logging.info(f"{label} - {method} - Max AUC: {max_auc:.05f}")
            logging.info(f"{label} - {method} - Min AUC: {min_auc:.05f}")
            print()

            # MLFlow
            mlflow.log_metric(f"{label} {cls_method} Mean AUC", avg_auc)
            # mlflow.log_metric(f"{label} {cls_method} Std AUC", med_auc)
    # Save AUC result
    return AUC_result
                


class IterativeEvaluation:
    def __init__(self, label_data: pd.DataFrame, testdata: pd.DataFrame, sample_list: list, methods_list: list):
        self.methods_list = methods_list
        self.label_data = label_data
        self.testdata = testdata
        self.sample_list = sample_list


    def classification(self, H):
        # Prepping the data and result
        logging.info("Starting iterative evaluation on classification")
        auc_columns = []
        for label in self.label_data.columns:
            for method in self.methods_list:
                auc_columns.append(f"{label}_{method}_AUC")
        AUC_result = pd.DataFrame(
            index = self.testdata.index,
            columns = auc_columns
        )

        # Assume label is 0-1
        for label in self.label_data.columns:
            # Iterate through each test
            for test_id in tqdm(self.testdata.index, desc=f"Evaluating label {label} on testdata"):
                # Get sample IDs
                train_sample_ids = self.testdata.loc[test_id, f'{label}_train']
                test_sample_ids = self.testdata.loc[test_id, f'{label}_test']

                # Get train test X/Y
                X_train = H.loc[train_sample_ids].values
                Y_train = self.label_data.loc[train_sample_ids, label]
                X_test = H.loc[test_sample_ids].values
                Y_test = self.label_data.loc[test_sample_ids, label]

                # Evaluate each method
                for cls_method in self.methods_list:
                    if(cls_method == "SVM"):                    cls = SVC(probability=True, verbose=False)
                    elif(cls_method == "Random Forest"):        cls = RandomForestClassifier(verbose=False)
                    elif(cls_method == "Logistic Regression"):  cls = LogisticRegression(max_iter=1000, verbose=False)
                    elif(cls_method == "AdaBoost"):             cls = AdaBoostClassifier()

                    # Fit the model
                    cls.fit(X_train, Y_train)
                    predicted = cls.predict_proba(X_test)[::,1]
                    fpr, tpr, _ = roc_curve(Y_test, predicted)
                    auc_val = auc(fpr, tpr)

                    # Store the result
                    AUC_result.at[test_id, f"{label}_{cls_method}_AUC"] = auc_val


        # Calculate, log statistics
        logging.info("Eval completed. Calculating statistics")
        results = {}
        for label in self.label_data.columns:
            for cls_method in self.methods_list:
                auc_values = AUC_result[f"{label}_{cls_method}_AUC"].values
                avg_auc = np.mean(auc_values)
                results[f"{label}_{cls_method}"] = avg_auc
        return results
    

    def evaluate(self, Ws, H, step):
        # CuPy Sanitize
        if isinstance(H, cp.ndarray): 
            H = H.get()
            Ws = [W.get() for W in Ws]


        logging.info("Starting evaluation")
        H_df = pd.DataFrame(H.T, index=self.sample_list, columns=[f"Latent_{i:03}" for i in range(H.shape[0])])
        
        # Classification
        result_cls = self.classification(H_df)
        for keyword, value in result_cls.items():
            data_pack = keyword.split("_")
            label = data_pack[0]
            method = data_pack[1]
            mlflow.log_metric(f"{label} {method} AUC over iterations", value, step=step)
            

