# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: main_ablation_studies.py
# Date: 2025/02/24 17:25:24
# Description: 
# 
# (c) 2025 bu1th4nh. All rights reserved. 
# Written with dedication in the University of Central Florida, EPCOT and the Magic Kingdom.
# -----------------------------------------------------------------------------------------------


import os
import json
import mlflow
import random
import pymongo
import logging
import multiprocessing

import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Any, Tuple, Union, Literal

def randomize_run_name():
    return f"{random.choice(royals_name)}_{random.choice(royals_name)}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

tqdm.pandas()
np.set_printoptions(edgeitems=20, linewidth=1000, formatter=dict(float=lambda x: "%.03f" % x))



if __name__ == '__main__':
    from env_config import *
    from log_config import initialize_logging
    from downstream.classification import evaluate_one_target

    # run_name = "overall_our_model_fixed_config"
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
    result_collection = mongo_db[mongo_collection]
    ablation_results = mongo_db['ABLATION_STUDIES']


    def find_run(run_id: str, target_id: str): return result_collection.find_one({'run_id': run_id, 'target_id': target_id})
    # -----------------------------------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------------------------------
    initialize_logging(log_filename = 'classification.log')
    logging.info(f"Starting classification evaluation on {args.run_mode} mode, storage mode {args.storage_mode}")




    # -----------------------------------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------------------------------
    # Get all runs and testdata
    logging.info("Fetching all runs configs and testdata")
    run_config_folders = [f's3://{a}' for a in s3.ls(RESULT_PRE_PATH)] if s3 is not None else os.listdir(RESULT_PRE_PATH)
    target_folders = [f's3://{a}' for a in s3.ls(TARG_PATH)] if s3 is not None else os.listdir(TARG_PATH)
    

    # Preload set
    run_cfg_data = {}
    tar_data = {}


    # Preload data
    logging.info("Preloading data")
    for cfg in tqdm(run_config_folders, desc='Preloading config'): 
        cfg_id = cfg.split('/')[-1]
        # print(cfg_id, cfg)
        if 'baseline' in cfg_id: continue
        try:
            run_cfg_data[cfg_id] = pd.read_parquet(f'{cfg}/H.parquet', storage_options=storage_options)
        except FileNotFoundError:
            logging.error(f"Config {cfg} not found. Skipping...")
    for tar in tqdm(target_folders, desc='Preloading target data'):
        target_id = str(tar.split('/')[-1]).split('.')[0]
        # print(target_id, tar)
        try:
            tar_data[target_id] = pd.read_parquet(tar, storage_options=storage_options)
        except FileNotFoundError:
            logging.error(f"Target {tar} not found. Skipping...")


    # -----------------------------------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------------------------------
    # Get results from all targets
    logging.info("Get results from all targets")
    Ariel = (
        result_collection
        .find(
            { "run_name": "overall_our_model_fixed_config" },
        ).to_list()
    )

    for run_package in Ariel:
        run_name = run_package['run_name']
        best_config = run_package['best_config']
        classifier = run_package['classifier']
        target_id = run_package['target_id']
        
        
        # Modify alpha
        half_before_alpha = best_config.split('-alpha-')[0]
        half_after_alpha = best_config.split('-alpha-')[1].split('-', maxsplit = 1)[1]

        max_alpha_cfg_id = f"{half_before_alpha}-alpha-10000-{half_after_alpha}"
        min_alpha_cfg_id = f"{half_before_alpha}-alpha-0-{half_after_alpha}"

        # Prep result
        min_alpha_abl_result = {
            'dataset': dataset_id,
            'target_id': target_id,
            'run_name': 'zero_alpha',
            'config': min_alpha_cfg_id,
            'classifier': classifier,
            "summary": {},
        }

        H = run_cfg_data[min_alpha_cfg_id]
        target = tar_data[target_id]
        min_alpha_abl_result["Overall"] = evaluate_one_target(H, target, [classifier], target_id)[classifier]


        # Compute summary
        metrics_list = ["AUROC", "ACC", "PRE", "REC", "F1", "MCC", "AUPRC"]
        Belle = pd.DataFrame.from_dict(min_alpha_abl_result['Overall'], orient='index')[metrics_list]
        for metric in metrics_list:
            min_alpha_abl_result['summary'][f'Mean {metric}'] = float(Belle[metric].mean(skipna=True))
            min_alpha_abl_result['summary'][f'Median {metric}'] = float(Belle[metric].median(skipna=True))
            min_alpha_abl_result['summary'][f'Std {metric}'] = float(Belle[metric].std(skipna=True))
            min_alpha_abl_result['summary'][f'Max {metric}'] = float(Belle[metric].max(skipna=True))
            min_alpha_abl_result['summary'][f'Min {metric}'] = float(Belle[metric].min(skipna=True))


        # Save to MLFlow
        logging.info(f"========================================================\n\n\n\n")
        logging.info(f"Summary for target {target_id}")
        for key in min_alpha_abl_result['summary'].keys():
            logging.info(f"{key}: {min_alpha_abl_result['summary'][key]}")

            if 'Mean AUROC' in key: 
                for method in classification_methods_list:
                    mlflow.log_metric(f'{target_id} Overall AUC', min_alpha_abl_result['summary'][key])
            if 'Mean MCC' in key: mlflow.log_metric(f'{target_id} Overall MCC', min_alpha_abl_result['summary'][key])


        # Save to MongoDB 
        ablation_results.insert_one(min_alpha_abl_result)
