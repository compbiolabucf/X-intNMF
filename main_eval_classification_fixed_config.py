# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: main_eval_classification.py
# Date: 2024/12/06 12:20:06
# Description: 
# 
# (c) 2024 bu1th4nh. All rights reserved. 
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

    run_name = "overall_our_model_fixed_config"
    # -----------------------------------------------------------------------------------------------
    # MongoDB
    # -----------------------------------------------------------------------------------------------
    mongo = pymongo.MongoClient(
        host='mongodb://localhost',
        port=27017,
        username='bu1th4nh',
        password='ariel.anna.elsa',
    )
    mongo_db = mongo['SimilarSampleCrossOmicNMF']
    eval_collection = mongo_db[mongo_collection]
    hparams_runs = mongo_db['HPARAMS_OPTS']


    def find_run(run_id: str, target_id: str): return eval_collection.find_one({'run_id': run_id, 'target_id': target_id})
    # -----------------------------------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------------------------------
    initialize_logging(log_filename = 'classification.log')
    logging.info(f"Starting classification evaluation on {args.run_mode} mode, storage mode {args.storage_mode}")


    # -----------------------------------------------------------------------------------------------
    # MLFlow
    # -----------------------------------------------------------------------------------------------
    mlflow.set_tracking_uri(uri="http://127.0.0.1:6969")
    mlflow.set_experiment(experiment_name)



    # -----------------------------------------------------------------------------------------------
    # Run ID Retrieval
    # -----------------------------------------------------------------------------------------------
    all_runs = mlflow.search_runs()[['run_id', 'tags.mlflow.runName']]
    possible_run_ids = all_runs[all_runs['tags.mlflow.runName'] == run_name]
    run_id = possible_run_ids['run_id'].values[0] if len(possible_run_ids) > 0 else None



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
        run_cfg_data[cfg_id] = pd.read_parquet(f'{cfg}/H.parquet', storage_options=storage_options)
    for tar in tqdm(target_folders, desc='Preloading target data'):
        target_id = str(tar.split('/')[-1]).split('.')[0]
        # print(target_id, tar)
        tar_data[target_id] = pd.read_parquet(tar, storage_options=storage_options)


    # -----------------------------------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------------------------------
    with mlflow.start_run(run_id=run_id) if run_id is not None else mlflow.start_run(run_name=run_name):
        if run_id is None: run_id = mlflow.active_run().info.run_id

        # Load and evaluate best config for each test
        for target_id in list(tar_data.keys()):
            # Pre-check
            update_mode = False
            if find_run(run_id, target_id) is not None: update_mode = True

            # Initialize
            result_for_target = {
                "run_id": run_id, # To be determined at MLFlow
                "target_id": target_id,
                "run_name": run_name,
                "summary": {},
                # "Overall": {},
            }

            # Load target data
            target = tar_data[target_id]
            Ariel = list(
                hparams_runs
                .find(
                    {
                        "dataset": dataset_id,
                        "target_id": target_id,
                    },
                    {
                        "_id": 0,
                        "test_id": 1,   
                        "config": 1,
                        "classifier": 1,
                        "AUROC": 1,
                    }
                )
            )
            Ariel = pd.DataFrame.from_records(Ariel)
            Ariel['config'] = Ariel[['config', 'classifier']].apply(lambda row: f"{row['config']}|{row['classifier']}", axis=1)
            Ariel = Ariel[['config', 'AUROC']].groupby('config').mean()
            best_cfg = Ariel.index[np.argmax(Ariel.values)]
            
            config_id = best_cfg.split('|')[0]
            classifier = best_cfg.split('|')[1]
            H = run_cfg_data[config_id]


            result_for_target["Overall"] = evaluate_one_target(H, target, [classifier], target_id)[classifier]


            

            # Evaluate
            # data_pack = evaluate_one_test(H, train_sample_ids, train_gnd_truth, test_sample_ids, test_gnd_truth, classifier)
            # metrics_list = [x for x in data_pack.keys() if str(x).isupper()] # Assume all metrics are uppercase

            # Build metadata
            result_for_target['best_AUROC_CV_train'] = np.max(Ariel.values)
            result_for_target['best_config'] = config_id
            result_for_target['classifier'] = classifier



            # Save to result
            


            # Compute summary
            metrics_list = ["AUROC", "ACC", "PRE", "REC", "F1", "MCC", "AUPRC"]
            Belle = pd.DataFrame.from_dict(result_for_target['Overall'], orient='index')[metrics_list]
            for metric in metrics_list:
                result_for_target['summary'][f'Mean {metric}'] = float(Belle[metric].mean(skipna=True))
                result_for_target['summary'][f'Median {metric}'] = float(Belle[metric].median(skipna=True))
                result_for_target['summary'][f'Std {metric}'] = float(Belle[metric].std(skipna=True))
                result_for_target['summary'][f'Max {metric}'] = float(Belle[metric].max(skipna=True))
                result_for_target['summary'][f'Min {metric}'] = float(Belle[metric].min(skipna=True))


            # Save to MLFlow
            logging.info(f"========================================================\n\n\n\n")
            logging.info(f"Summary for target {target_id}")
            for key in result_for_target['summary'].keys():
                logging.info(f"{key}: {result_for_target['summary'][key]}")

                if 'Mean AUROC' in key: 
                    for method in classification_methods_list:
                        mlflow.log_metric(f'{target_id} Overall AUC', result_for_target['summary'][key])
                if 'Mean MCC' in key: mlflow.log_metric(f'{target_id} Overall MCC', result_for_target['summary'][key])


            # Save to MongoDB
            if update_mode: eval_collection.update_one(
                {'run_id': run_id, 'target_id': target_id},
                {'$set': result_for_target}
            )
            else: eval_collection.insert_one(result_for_target)
