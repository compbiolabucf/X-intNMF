# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: main_dim_reduction.py
# Date: 2024/12/01 17:16:57
# Description: Main script for running hyperparams dim reduction for SimilarSampleCrossOmicNMF
# 
# (c) 2024 bu1th4nh. All rights reserved. 
# Written with dedication in the University of Central Florida, EPCOT and the Magic Kingdom.
# -----------------------------------------------------------------------------------------------



import os
import mlflow
import random
import logging
import warnings
import cupy as cp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from tqdm import tqdm
warnings.filterwarnings("ignore")


from env_config import *
from model.crossOmicNMF import SimilarSampleCrossOmicNMF
from downstream.checkpoint_utils import IterativeCheckpointing


def randomize_run_name():
    return f"{random.choice(royals_name)}_{random.choice(royals_name)}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

cp.cuda.runtime.setDevice(1)
tqdm.pandas()
np.set_printoptions(edgeitems=20, linewidth=1000, formatter=dict(float=lambda x: "%.03f" % x))


# -----------------------------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------------------------
from log_config import initialize_logging
initialize_logging(None)


# -----------------------------------------------------------------------------------------------
# MLFlow
# -----------------------------------------------------------------------------------------------
import mlflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:6969")
mlflow.set_experiment(experiment_name)


# -----------------------------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------------------------
bipart_data = pd.read_parquet(f'{DATA_PATH}/bipart.parquet', storage_options=storage_options)
miRNA = pd.read_parquet(f'{DATA_PATH}/miRNA.parquet', storage_options=storage_options)
mRNA = pd.read_parquet(f'{DATA_PATH}/mRNA.parquet', storage_options=storage_options)

features_list = [mRNA.index.to_list(), miRNA.index.to_list()]   
sample_list = mRNA.columns.to_list()


omics_data = [mRNA.to_numpy(np.float64, True), miRNA.to_numpy(np.float64, True)]
off_diag_interactions = {(0, 1): bipart_data.to_numpy(np.float64, True)}
m = [omic.shape[0] for omic in omics_data]




# -----------------------------------------------------------------------------------------------
# Hyperparams DR runs
# -----------------------------------------------------------------------------------------------
logging.warning("Running hyperparams runs")
latent_possibilities = [10, 25, 50, 100, 200]
alpha_possibilities = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 0]
beta_possibilities = [1, 0.1, 0.01, 10, 100]


prioritized_runs = [
    (10, 0.0001, 100), 
    (10, 0.001, 0.1), 
    (10, 0.1, 0.01), 
    (10, 0, 0.01), 
    (10, 1, 100), 
    (10, 100, 100), 
    (10, 1000, 100), 
    (10, 10000, 10), 
    (25, 0.01, 0.01), 
    (25, 0.01, 100), 
    (50, 0.001, 1), 
    (50, 0.001, 10), 
    (50, 10000, 10), 
    (100, 0.0001, 0.01), 
    (100, 0.01, 0.01), 
    (100, 0.01, 100), 
    (100, 10, 100), 
    (100, 1000, 0.1), 
    (100, 1000, 10), 
    (200, 0, 10), 
    (200, 0.0001, 10), 
    (200, 0.001, 0.1), 
    (200, 0.1, 0.01), 
    (200, 1, 0.1), 
]


# Prepare running pack
running_pack = []
for latent_size in latent_possibilities:
    for beta in beta_possibilities:
        for alpha in alpha_possibilities:
            running_pack.append((latent_size, alpha, beta))
random.shuffle(running_pack)
running_pack = prioritized_runs + running_pack

# Run
for latent_size, alpha, beta in running_pack[2:]:
    latent_columns = [f"latent_{i:03}" for i in range(latent_size)]
    logging.info(f"Running: alpha = {alpha}, beta = {beta}")
    run_name = f"k-{latent_size}-alpha-{alpha}-beta-{beta}-gamma-overridden"
    result_path = f"{RESULT_PRE_PATH}/{run_name}"


    # Pickup if already exists. Update for S3
    if storage_options is not None:
        if s3.exists(f'{result_path}/H.parquet') and pickup_leftoff_mode: 
            logging.warning(f"Result path {result_path} already exists. Skipping...")
            continue
    else:
        if os.path.exists(f'{result_path}/H.parquet') and pickup_leftoff_mode: 
            logging.warning(f"Result path {result_path} already exists. Skipping...")
            continue


    # Checkpointing
    if storage_options is not None: s3.makedirs(f"{result_path}/checkpoints", exist_ok=True)
    else: os.makedirs(f"{result_path}/checkpoints", exist_ok=True)
    drc_saving = IterativeCheckpointing(
        sample_list = sample_list,
        omics_features = features_list,
        prefix_path = result_path,
        storage_options = storage_options,
        s3 = s3
    )


    # Run
    try:
        with mlflow.start_run(run_name = run_name):
            MODEL = SimilarSampleCrossOmicNMF(
                omics_layers=omics_data,
                cross_omics_interaction=off_diag_interactions,
                k=latent_size,
                alpha=alpha,
                betas=beta,
                gammas=1,
                max_iter= 10000 if latent_size < 100 else 5000,
                tol=1e-4,
                verbose=True
            )

            # try:
            new_wds, H = MODEL.solve(run_mode='full', use_cross_validation=True, additional_tasks=[drc_saving.save])
            # except Exception as e:
            #     logging.error(f"Error occurred: {e}")
            #     raise e
                
            
            
            logging.info(f"Saved hyperparams alpha = {alpha}, beta = {beta}...")
            drc_saving.save(new_wds, H, step=None)


            # Save Run ID to file
            run_id = mlflow.active_run().info.run_id
            if storage_options is not None:
                with s3.open(f"{result_path}/run_id.txt", "w") as f:
                    f.write(run_id)
            else:
                with open(f"{result_path}/run_id.txt", "w") as f:
                    f.write(run_id)


    except Exception as e:
        logging.error(f"Error occurred during run: {e}")
        # raise e
        continue

        

# Content and code by bu1th4nh. Written with dedication in the University of Central Florida and the Magic Kingdom.
# Powered, inspired and motivated by EDM, Counter-Strike and Disney Princesses. 
# Image credit: https://emojicombos.com/little-mermaid-text-art
#                                                                                                           
#                                                          ⡀⣰    
#                                                         ⣰⡿⠃    
#                                                        ⣼⣿⣧⠏    
#                                                       ⣰⣿⠟⠋     
#                                                       ⣿⡿       
#                                                      ⣸⣿⡇       
#                                        ⣀⣴⣾⣿         ⢰⣿⣿⡇       
#                                    ⢀⣠⣾⣿⣿⣿⣿⡏         ⣼⣿⣿        
#                                   ⣠⣿⣿⣿⣿⣿⣿⣿⣤        ⢠⣿⣿⠇        
#                                  ⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⡄    ⢀⣾⣿⣿         
#                                 ⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠄  ⣤⣾⣿⡟⠁         
#                                ⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⠁⢀⣴⣾⣿⣿⠏           
#                             ⣀⣀⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⣴⣿⣿⣿⠿⠁            
#                       ⣀⣤⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏              
#                     ⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣄              
#                    ⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡄            
#                    ⣾⣿⣿⣿⠿⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇            
#                   ⣸⣿⡿⠉⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠉⢿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠁            
#               ⠠⣤⣴⣾⡿⠋ ⣼⣿⣿⣿⢟⣿⣿⠏⢰⣿⡿⠟⢻⣿⣿⡿   ⠈⢿⣿⣿⣿⣿⣿⣿⠏              
#                     ⣼⣿⣿⡟⠁⣸⡿⠁ ⠘⠋⣀⣴⣿⣿⠟    ⢀⣼⣿⣿⣿⣿⣿⠃               
#                  ⢠⣴⣾⣿⠿⠋ ⠐⠋   ⢀⣾⣿⣿⡿⠋   ⣠⣴⣾⣿⣿⣿⣿⣿⠃                
#                            ⢀⣴⣿⡿⠛⠉   ⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿                 
#                          ⢠⣶⡿⠋⠁    ⢠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡆                
#                         ⣠⣿⣿⡇     ⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇                
#                       ⠴⣿⣿⠋⡿⠁    ⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠁                
#                        ⠿⠏       ⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏                 
#                                ⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟                  
#                                ⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟                   
#                               ⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠋                    
#                               ⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠁                     
#                             ⢀⣾⣿⣿⣿⣿⣿⣿⣿⡿⠟                        
#                           ⣀⣴⣿⣿⣿⣿⣿⣿⣿⠿⠋                          
#                        ⣀⣤⣾⣿⣿⣿⣿⣿⡿⠟⠋                             
#               ⣀⣀⣀⣠⣤⣤⣤⣶⣿⣿⣿⣿⡿⠿⠛⠋⠁                                
#         ⢀⣠⣶⣶⣿⣿⣿⣿⣿⣿⣿⣿⠟⠛⠋⠉⠉                                      
#       ⢀⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁                                           
#      ⣠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇                                            
#     ⢠⣿⣿⣿⣿⣿⣿⡿⠟⠋⢠⣿⣿⣿⣿⣧                                           
#     ⣼⣿⡿⠟⠋⠉    ⠸⣿⣿⣿⣿⣿⣧                                          
#     ⣿⡟         ⣿⣿⣿⣿⣿⣿                                          
#     ⠸⠇         ⣿⣿⣿⣿⣿⣿                                          
#                ⢸⣿⣿⣿⣿⡟                                          
#                ⣼⣿⣿⣿⠟                                           
#               ⢀⣿⡿⠛⠁                                            
#            