# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: main_run_params_optimization.py
# Date: 2024/12/13 15:21:11
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

tqdm.pandas()

def randomize_run_name():
    return f"{random.choice(royals_name)}_{random.choice(royals_name)}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

tqdm.pandas()
np.set_printoptions(edgeitems=20, linewidth=1000, formatter=dict(float=lambda x: "%.03f" % x))



if __name__ == '__main__':
    from env_config import *
    from log_config import initialize_logging
    from downstream.classification import evaluate_one_target, hparams_evaluate, HParamsParallelWrapper


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
    collection = mongo_db['HPARAMS_OPTS']


    def find_run(dataset_id: str, target_id: str, test_id: str, config: str, classifier: str): return collection.find_one({
        'dataset': dataset_id, 
        'target_id': target_id, 
        'test_id': test_id,
        'config': config,
        'classifier': classifier,
    })
    # -----------------------------------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------------------------------
    initialize_logging(log_filename = 'classification.log')
    logging.info(f"Starting classification evaluation on {args.run_mode} mode, storage mode {args.storage_mode}")



    # -----------------------------------------------------------------------------------------------
    # Classification
    # -----------------------------------------------------------------------------------------------
    # Get all runs and testdata
    logging.info("Fetching all runs and testdata")
    run_config_folders = [f's3://{a}' for a in s3.ls(RUN_CFG_PATH)] if s3 is not None else os.listdir(RUN_CFG_PATH)
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



    # Load to queue
    run_queue = []
    logging.info("Loading to queue")
    for target_id in list(tar_data.keys()):
        target = tar_data[target_id]
        for test_id in tqdm(target.index, desc = f'Loading {target_id} to queue'):
            for cfg_id in run_cfg_data.keys():
                for classifier in classification_methods_list:
                    if find_run(dataset_id, target_id, test_id, cfg_id, classifier) is not None: continue
                    run_queue.append({
                        'target_id': target_id,
                        'test_id': test_id,
                        'config': cfg_id,
                        'classifier': classifier,
                    })
    
    logging.info(f"Loaded {len(run_queue)} runs to queue")

            
    
    # Run the classification
    logging.info("Starting classification")
    wrapper = HParamsParallelWrapper(run_cfg_data, tar_data, dataset_id)
    if args.parallel:
        with multiprocessing.Pool(processes=10) as pool:
            results = pool.starmap(wrapper, run_queue)
    else:   
        # results = []
        results = pd.Series(run_queue).progress_apply(wrapper.eval_nonparallel).tolist()
        # for run in tqdm(run_queue, desc="Running classification"): 
    logging.info("Classification done, saving results")



    # Save the results
    collection.insert_many(results)
