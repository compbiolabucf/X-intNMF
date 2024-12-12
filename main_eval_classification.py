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
    collection = mongo_db[str(args.run_mode).upper()]


    def find_run(run_id: str, target_id: str): return collection.find_one({'run_id': run_id, 'target_id': target_id})
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
    # Parallelize
    # -----------------------------------------------------------------------------------------------
    def process_run(run_info, run_data, tar_data):
        # Unpack
        run_id = run_info['run_id']
        run_folder = run_info['run_folder']
        target_id = run_info['target_id']

        # Load the data
        H = pd.DataFrame.from_dict(run_data[run_folder], orient='index')
        test_data = pd.DataFrame.from_dict(tar_data[target_id], orient='index')

        # Evaluate
        result_pack = evaluate_one_target(H, testdata = test_data, methods_list = classification_methods_list, target = target_id)

        # Load to staging package
        data_pack = {
            'run_id': run_id,
            'target_id': target_id,
            'summary': {}
        }
        for method in result_pack.keys():
            data_pack[method] = result_pack[method].to_dict(orient='index')

            for metric in result_pack[method].columns:
                if str(metric).isupper():
                    data_pack['summary'][method] = {
                        f'Mean {metric}': float(np.mean(result_pack[method][metric].values)),
                        f'Median {metric}': float(np.median(result_pack[method][metric].values)),
                        f'Std {metric}': float(np.std(result_pack[method][metric].values)),
                        f'Max {metric}': float(np.max(result_pack[method][metric].values)),
                        f'Min {metric}': float(np.min(result_pack[method][metric].values)),
                    }

            
        # MLFlow
        # with mlflow.start_run(run_name = randomize_run_name()):
        with mlflow.start_run(run_id=run_id):
            run_name = mlflow.active_run().info.run_name
            data_pack['run_name'] = run_name

            for key in data_pack['summary'].keys():
                # if 'Mean AUROC' in key: mlflow.log_metric(f'{target_id} {key}', data_pack['summary'][key])
                if 'Mean MCC' in key: mlflow.log_metric(f'{target_id} {key}', data_pack['summary'][key])


        # Save to MongoDB
        collection.insert_one(data_pack)


    # -----------------------------------------------------------------------------------------------
    # Classification
    # -----------------------------------------------------------------------------------------------
    patience_delay = 0
    while True:
        # Get all runs and testdata
        logging.info("Fetching all runs and testdata")
        run_folders = [f's3://{a}' for a in s3.ls(RESULT_PRE_PATH)] if s3 is not None else os.listdir(RESULT_PRE_PATH)
        target_folders = [f's3://{a}' for a in s3.ls(TARG_PATH)] if s3 is not None else os.listdir(TARG_PATH)
        run_queue = []

        # Preload set
        actual_runs = set()
        actual_targets = set()
        
        # Iterate through each run & folders, and check if the run is already in the database
        # If not, add to the queue
        for run_folder in run_folders:
            run_id = s3.open(f'{run_folder}/run_id.txt', 'r').read() if s3 is not None else open(f'{run_folder}/run_id.txt', 'r').read()

            # Special case, TODO workarounds: Skip baselines
            if str(run_folder.split('/')[-1]).startswith('baseline'): continue

            # Check if run already exists
            for target_folder in target_folders:
                target_id = str(target_folder.split('/')[-1]).split('.')[0]
                if find_run(run_id, target_id) is not None:
                    logging.info(f"Run {run_id} on dataset {target_id} already exists. Skipping")
                    continue
                run_queue.append({
                    'run_id': run_id,
                    'run_folder': run_folder,
                    'target_id': target_id,
                })
                actual_runs.add(run_folder)
                actual_targets.add(target_id)

        # If no new runs, increase patience
        logging.info(f"Found {len(run_queue)} new runs to evaluate")
        if len(run_queue) == 0:
            patience_delay += 1
            if patience_delay == 5: break
            else: 
                os.sleep(60)
                continue

        # Reset patience
        patience_delay = 0


        # Preload the data
        logging.info("Preloading data")
        run_data = {run: pd.read_parquet(f'{run}/H.parquet', storage_options=storage_options).to_dict(orient='index') for run in actual_runs}
        tar_data = {target: pd.read_parquet(f'{TARG_PATH}/{target}.parquet', storage_options=storage_options).to_dict(orient='index') for target in actual_targets}
        

        # Iterate through each run
        if args.parallel:
            logging.info("Starting parallel classification evaluation")

            parallel_args = [(run_info, run_data, tar_data) for run_info in run_queue]
            with multiprocessing.Pool(processes=24) as pool:
                tqdm(pool.starmap(process_run, parallel_args))
        else:
            logging.info("Starting sequential classification evaluation")
            for run_info in tqdm(run_queue):
                process_run(run_info, run_data, tar_data)
        

        # Cleanup
        logging.info("Cleaning up")
        del run_data
        del tar_data
        del run_queue
        del actual_runs
        del actual_targets
