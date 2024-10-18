# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành (@bu1th4nh)
# Title: classification.py
# Date: 2024/10/03 15:22:17
# Description: 
# 
# (c) bu1th4nh. All rights reserved
# -----------------------------------------------------------------------------------------------


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import logging
import random
import json
import os
warnings.filterwarnings("ignore")


from tqdm import tqdm

import mlflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:6969")
mlflow.set_experiment("SimilarSampleCrossOmicNMF")


from log_config import initialize_logging
initialize_logging()


from config import *
from evaluation import evaluate
from ml import test, dataset_prep, core_data_init







if __name__ == '__main__':
    # Run all methods
    positive_samples, negative_samples, clinical = core_data_init()

    if (not os.path.exists("attempts.json")):
        attempts = []
        for Ariel in tqdm(range(TRAIN_TIME), desc="Running"):
            # Prepare dataset
            attempt = {} # run_id -> 
            train_idx, test_idx = dataset_prep(positive_samples, negative_samples)

            # Evaluate each run preset
            for run_preset in run_presets:
                run_id = run_preset['run_id']
                run_name = run_preset['run_name']
                logging.info(f"Running {run_name}...")
                attempt[run_id] = test(
                    run_name=run_name, 
                    train_id=train_idx, 
                    test_id=test_idx, 
                    clinical=clinical.copy(deep=True)
                )

            # Save attempt
            attempts.append(attempt)
            
        # Save attempts for further evaluation
        if not DISABLE_FILE:
            with open(f'attempts.json', 'w') as f:
                json.dump(attempts, f)

    
    else: attempts = json.loads(open('attempts.json').read())

    # Evaluate
    print("\n\n")
    logging.warning("Evaluating")
    # Get mean AUC by each method and each run preset
    for run_preset in run_presets:
        logging.info(f"Evaluating {run_preset['run_name']}...")
        run_id = run_preset['run_id']
        if DISABLE_MLFLOW:
            evaluate(attempts, run_preset)
        else:
            with mlflow.start_run(run_id=run_id):
                evaluate(attempts, run_preset)




    