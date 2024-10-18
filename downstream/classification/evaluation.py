# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành (@bu1th4nh)
# Title: evaluation.py
# Date: 2024/10/07 23:08:01
# Description: 
# 
# (c) bu1th4nh. All rights reserved
# -----------------------------------------------------------------------------------------------


import mlflow
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc



from config import *

def evaluate(attempts, run_preset):
    # -----------------------------------------------------------------------------------------------
    # Build results
    # -----------------------------------------------------------------------------------------------
    run_id = run_preset['run_id']
    run_name = run_preset['run_name']
    preset_results_auc = {
        "AdaBoost"              : [],
        "Logistic Regression"   : [],
        "Random Forest"         : [],
        "SVM"                   : [],
    }

    # Attempts: List[Dict[Preset, Dict[Method, Dict[Metrics, Value]]]]
    for attempt in attempts:
        preset = attempt[run_id]
        for method in METHODS:
            preset_results_auc[method].append(preset[method]['auc'])
            

    # -----------------------------------------------------------------------------------------------
    # Plot & arrays intialization
    # -----------------------------------------------------------------------------------------------
    # AUC Histogram
    auc_hist_fig, auc_hist_ax = plt.subplots(2, 2, figsize=(20, 7))
    auc_hist_fig.suptitle(f"AUC Hist - {len(attempts)} pass")

    # AUC Boxplot
    combined_auc_data = []
    combined_auc_labels = []
    auc_box_fig, auc_box_ax = plt.subplots(figsize=(20, 7))
    auc_box_ax.set_title(f"Boxplot AUC - {len(attempts)} pass")

    # Nearest ROC Curve
    auc_nr_fig, auc_nr_ax = plt.subplots(2, 2, figsize=(20, 7))
    auc_nr_fig.suptitle(f"Nearest ROC Curve - {len(attempts)} pass")


    # -----------------------------------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------------------------------
    for meth_id, cls_method in enumerate(METHODS):
        # Descriptive statistics
        aucs = preset_results_auc[cls_method]
        avg_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        max_auc = np.max(aucs)
        min_auc = np.min(aucs)
        med_auc = np.median(aucs)
        nearest_auc = np.argmin(np.abs(aucs - avg_auc))


        # Logging
        logging.info(f"Method: {cls_method}")
        logging.info(f"Mean AUC: {avg_auc}")
        logging.info(f"Median AUC: {med_auc}")
        logging.info(f"Std AUC: {std_auc}")
        logging.info(f"Max AUC: {max_auc}")
        logging.info(f"Min AUC: {min_auc}")
        if not DISABLE_MLFLOW: mlflow.log_metric(f"{cls_method} Mean AUC {TRAIN_TIME} pass", avg_auc)



        # Histogram
        axxx = auc_hist_ax[meth_id//2, meth_id%2]
        sns.histplot(preset_results_auc[method], kde=True, ax=axxx)
        axxx.set_title(f'{method} AUC = {avg_auc:.4f} ± {std_auc:.4f} [{min_auc:.4f}, {max_auc:.4f}]')


        # Nearest ROC Curve
        axxx = auc_nr_ax[meth_id//2, meth_id%2]
        tpr = attempts[nearest_auc][run_id][method]['tpr']
        fpr = attempts[nearest_auc][run_id][method]['fpr']
        axxx.plot(fpr, tpr)
        axxx.set_title(f'{method} AUC = {avg_auc:.4f} @ attempt #[{nearest_auc}]')


        # Boxplot
        combined_auc_data += aucs
        combined_auc_labels += [f"{cls_method} | median={med_auc:.4f}"] * len(aucs)
    sns.boxplot(x=combined_auc_labels, y=combined_auc_data, ax=auc_box_ax)

    # -----------------------------------------------------------------------------------------------
    # Show
    # -----------------------------------------------------------------------------------------------
    auc_hist_fig.show()
    auc_box_fig.show()
    auc_nr_fig.show()

    # -----------------------------------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------------------------------
    if not DISABLE_FILE:
        auc_hist_fig.savefig(f"{RESULT_PATH}/{run_name}/auc_hist_{TRAIN_TIME}pass.pdf")
        auc_box_fig.savefig(f"{RESULT_PATH}/{run_name}/auc_box_{TRAIN_TIME}pass.pdf")
        auc_nr_fig.savefig(f"{RESULT_PATH}/{run_name}/auc_nr_{TRAIN_TIME}pass.pdf")

        if not DISABLE_MLFLOW:
            mlflow.log_artifact(f"{RESULT_PATH}/{run_name}/auc_hist_{TRAIN_TIME}pass.pdf")
            mlflow.log_artifact(f"{RESULT_PATH}/{run_name}/auc_box_{TRAIN_TIME}pass.pdf")
            mlflow.log_artifact(f"{RESULT_PATH}/{run_name}/auc_nr_{TRAIN_TIME}pass.pdf")
        
 