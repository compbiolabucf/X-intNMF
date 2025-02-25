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
    # mlflow.set_tracking_uri(uri="http://127.0.0.1:6969")
    # mlflow.set_experiment(experiment_name)



    # -----------------------------------------------------------------------------------------------
    # Run ID Retrieval
    # -----------------------------------------------------------------------------------------------
    # all_runs = mlflow.search_runs()[['run_id', 'tags.mlflow.runName']]
    # possible_run_ids = all_runs[all_runs['tags.mlflow.runName'] == run_name]
    # run_id = possible_run_ids['run_id'].values[0] if len(possible_run_ids) > 0 else None



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