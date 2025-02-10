# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: main_eval_survival_fixed_config.py
# Date: 2025/02/06 16:12:36
# Description: 
# 
# (c) 2025 bu1th4nh. All rights reserved. 
# Written with dedication in the University of Central Florida, EPCOT and the Magic Kingdom.
# -----------------------------------------------------------------------------------------------


import os
import warnings 
warnings.filterwarnings("ignore") 
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Union, Literal

import pymongo
from s3fs import S3FileSystem

from sklearn import set_config
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import FitFailedWarning

from sksurv.linear_model import CoxnetSurvivalAnalysis
import lifelines

import matplotlib.pyplot as plt


if __name__ == '__main__':
    from env_config import *
    from log_config import initialize_logging
    from downstream.survival import surv_analysis


    # -----------------------------------------------------------------------------------------------
    # MongoDB
    # -----------------------------------------------------------------------------------------------
    mongo = pymongo.MongoClient(
        host='mongodb://localhost',
        port=27017,
        username='bu1th4nh',
        password='ariel.anna.elsa',
    )
    mongo_db = mongo[mongo_db_name]
    hparams_runs = mongo_db['HPARAMS_OPTS']
    surv_collection = mongo_db['SURVIVAL_ANALYSIS']



    # -----------------------------------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------------------------------
    initialize_logging(log_filename = 'classification.log')
    logging.info(f"Starting classification evaluation on {args.run_mode} mode, storage mode {args.storage_mode}")


    # -----------------------------------------------------------------------------------------------
    # MLFlow
    # -----------------------------------------------------------------------------------------------
    # mlflow.set_tracking_uri(uri="http://127.0.0.1:6969")
    # mlflow.set_experiment(experiment_name)



    # -----------------------------------------------------------------------------------------------
    # Run ID Retrieval
    # -----------------------------------------------------------------------------------------------
    # all_runs = mlflow.search_runs()[['run_id', 'tags.mlflow.runName']]
    # possible_run_ids = all_runs[all_runs['tags.mlflow.runName'] == run_name]
    # run_id = possible_run_ids['run_id'].values[0] if len(possible_run_ids) > 0 else None



    # -----------------------------------------------------------------------------------------------
    # Obtain survival analysis targets
    # -----------------------------------------------------------------------------------------------
    logging.info(f"Retrieving survival analysis targets from {SA_TARG_PATH}")
    surv_targets_data = {}
    surv_target_folder = [f's3://{a}' for a in s3.ls(SA_TARG_PATH)]
    for tar in tqdm(surv_target_folder, desc='Preloading target data'):
        target_id = str(tar.split('/')[-1]).split('.')[0]
        # print(target_id, tar)
        try:
            surv_targets_data[target_id] = pd.read_parquet(tar, storage_options=storage_options)
            print(surv_targets_data[target_id].columns)

        except FileNotFoundError:
            logging.error(f"Target {tar} not found. Skipping...")   



    # -----------------------------------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------------------------------
    # Get target id for each disease => find the best hparams for each target
    logging.info(f"Acquiring classification target ids for {dataset_id} for config")
    classification_target_ids_for_disease = hparams_runs.find(
        {'dataset': dataset_id},
    ).distinct('target_id')


    # Get the best hparams for each target and run SA
    logging.info(f"Acquiring classification target ids for {dataset_id} for config")
    for classification_target_id in classification_target_ids_for_disease:
        Ariel = (
            pd.DataFrame.from_records(
                hparams_runs
                .find(
                    {
                        "dataset": dataset_id,
                        "target_id": classification_target_id,
                    },
                    {
                        "_id": 0,
                        "test_id": 1,   
                        "config": 1,
                        "AUROC": 1,
                    }
                ).to_list()
            )[['config', 'AUROC']]
            .groupby('config')
            .mean()
        )
        best_cfg = Ariel.index[np.argmax(Ariel.values)]
        H = pd.read_parquet(f'{RUN_CFG_PATH}/{best_cfg}/H.parquet', storage_options=storage_options)


        # Get all survival analysis targets
        for surv_target_id in surv_targets_data.keys():
            survival = surv_targets_data[surv_target_id]
            if surv_target_id == 'survival': 
                event_label = 'Overall Survival Status'
                time_label = 'Overall Survival (Months)'
            else:
                event_label = 'Disease Free Status'
                time_label = 'Disease Free (Months)'

            attempt = 0
            while True:
                attempt += 1
                logging.info(f'SA for {dataset_id} with {classification_target_id}, config {best_cfg} and {surv_target_id}, attempt {attempt}')

                train_sample_ids, test_sample_ids = train_test_split(list(survival.index), test_size=0.2)
                try:
                    surv_result = surv_analysis(
                        H,
                        train_sample_ids,
                        survival.loc[train_sample_ids],
                        test_sample_ids,
                        survival.loc[test_sample_ids],
                        event_label,
                        time_label,
                    )


                    if surv_result['p_value'] < 0.05:
                        logging.info(surv_result)
                        break
                    else:
                        logging.info(f'p-value > 0.05: {surv_result["p_value"]}')
                except Exception as e:
                    logging.error('Error occurred:', e)
                    continue

            
            # Save result
            if surv_result is None:
                continue
            surv_result['dataset_id'] = dataset_id
            surv_result['surv_target'] = surv_target_id
            surv_result['best_cfg_from'] = classification_target_id
            surv_result['best_cfg'] = best_cfg
            
            surv_collection.insert_one(surv_result)
            logging.info(f"Saved result for {dataset_id} with {classification_target_id}, config {best_cfg} and {surv_target_id}")