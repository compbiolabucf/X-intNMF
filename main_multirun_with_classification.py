# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành (@bu1th4nh)
# Title: main_multirun.py
# Date: 2024/10/19 16:22:44
# Description: 
# 
# (c) bu1th4nh. All rights reserved
# -----------------------------------------------------------------------------------------------


import os
import mlflow
import random
import logging
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from tqdm import tqdm
warnings.filterwarnings("ignore")

from env_config import *
from model.crossOmicNMF import SimilarSampleCrossOmicNMF
from downstream.classification.evaluation_bulk import evaluate, IterativeEvaluation


def randomize_run_name():
    return f"{random.choice(royals_name)}_{random.choice(royals_name)}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"


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
testdata = pd.read_parquet(f'{DATA_PATH}/testdata_classification.parquet')
bipart_data = pd.read_parquet(f'{DATA_PATH}/bipart.parquet')
clinical = pd.read_parquet(f'{DATA_PATH}/clinical.parquet')
miRNA = pd.read_parquet(f'{DATA_PATH}/miRNA.parquet')
mRNA = pd.read_parquet(f'{DATA_PATH}/mRNA.parquet')

features_list = [mRNA.index.to_list(), miRNA.index.to_list()]   
sample_list = mRNA.columns.to_list()


omics_data = [mRNA.to_numpy(np.float64, True), miRNA.to_numpy(np.float64, True)]
off_diag_interactions = {(0, 1): bipart_data.to_numpy(np.float64, True)}
m = [omic.shape[0] for omic in omics_data]


for label in clinical: clinical[label] = clinical[label].apply(lambda x: 1 if x == 'Positive' else 0)
# -----------------------------------------------------------------------------------------------
# Baseline
# -----------------------------------------------------------------------------------------------
logging.warning("Establishing baseline")
for run_mode in ['raw_baseline', 'norm_baseline', 'nmf_lasso_only']:
    logging.info(f"Run mode: {run_mode}")
    baseline_columns = features_list[0] + features_list[1] if run_mode == 'raw_baseline' else [f"latent_{i:03}" for i in range(10)]
    run_name = f"baseline-{run_mode}"
    result_path = f"{result_pre_path}/{run_name}"
    

    # Pickup if already exists
    if os.path.exists(result_path) and pickup_leftoff_mode: continue

    # Run
    with mlflow.start_run(run_name = run_name):
        base = SimilarSampleCrossOmicNMF(
            omics_layers=omics_data,
            cross_omics_interaction=off_diag_interactions,
            k=10,
            alpha=1,
            betas=1,
            gammas=1,
            max_iter=20000,
            tol=1e-4,
            verbose=True
        )
        _, H = base.solve(run_mode=run_mode, use_cross_validation=True)
        
        
        # Save result
        os.makedirs(result_path, exist_ok=True)
        H_df = pd.DataFrame(H.T, index=sample_list, columns=baseline_columns)
        H_df.to_parquet(f"{result_path}/H.parquet")
        logging.info(f"Saved baseline {run_mode} to parquet file: {result_path}/H.parquet")

        
        # Evaluate
        auc_result_data = evaluate(H_df, clinical, testdata, classification_methods_list)
        auc_result_data.to_parquet(f"{result_path}/classification_result.parquet")
    

# -----------------------------------------------------------------------------------------------
# Hyperparams evaluation over iterations
# -----------------------------------------------------------------------------------------------
evaluator = IterativeEvaluation(clinical.copy(deep=True), testdata.copy(deep=True), sample_list, ['Logistic Regression'])


# -----------------------------------------------------------------------------------------------
# Hyperparams runs
# -----------------------------------------------------------------------------------------------
logging.warning("Running hyperparams runs")
alpha_possibilities = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 0]
beta_possibilities = [1, 0.1, 0.01, 10, 100]
latent_possibilities = [10, 25, 50, 100, 200]

# Prepare running pack
running_pack = []
for latent_size in latent_possibilities:
    for beta in beta_possibilities:
        for alpha in alpha_possibilities:
            running_pack.append((latent_size, alpha, beta))
random.shuffle(running_pack)


# Run
for latent_size, alpha, beta in running_pack:
    latent_columns = [f"latent_{i:03}" for i in range(latent_size)]
    logging.info(f"Running: alpha = {alpha}, beta = {beta}")
    run_name = f"k-{latent_size}-alpha-{alpha}-beta-{beta}-gamma-overridden"
    result_path = f"{result_pre_path}/{run_name}"
    
    # Pickup if already exists
    if os.path.exists(result_path) and pickup_leftoff_mode: 
        logging.warning(f"Result path {result_path} already exists. Skipping...")
        continue

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
            new_wds, H = MODEL.solve(run_mode='full', use_cross_validation=True, evaluation_metric=evaluator.evaluate)
            # except Exception as e:
            #     logging.error(f"Error occurred: {e}")
            #     raise e
                
            
            os.makedirs(result_path, exist_ok=True)
            H_df = pd.DataFrame(H.T, index=sample_list, columns=latent_columns)
            H_df.to_parquet(f"{result_path}/H.parquet")
            logging.info(f"Saved hyperparams alpha = {alpha}, beta = {beta} to parquet file: {result_path}/H.parquet")

            for d, Wd in enumerate(new_wds):
                pd.DataFrame(Wd, index=features_list[d], columns=latent_columns).to_parquet(f"{result_path}/W{d}.parquet")
                logging.info(f"Saved W{d} to parquet file: {result_path}/W{d}.parquet")

            # Evaluate
            logging.info(f"Evaluating hyperparams alpha = {alpha}, beta = {beta}")
            auc_result_data = evaluate(H_df, clinical, testdata, classification_methods_list)
            auc_result_data.to_parquet(f"{result_path}/classification_result.parquet")
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