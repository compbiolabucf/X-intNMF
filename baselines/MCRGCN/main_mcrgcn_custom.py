# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Main Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: main_mcrgcn_custom.py
# Date: 2024/11/23 20:44:50
# Description: 
#  - Main script for MCRGCN evaluation on custom dataset
#  - This script is adapted from the original MCRGCN implementation
# 
# (c) 2024 original authors. All rights reserved. 
# Written with dedication in the University of Central Florida, EPCOT and the Magic Kingdom.
# -----------------------------------------------------------------------------------------------


import os
import sys
import torch
import mlflow
import logging
import random
import pymongo
import numpy as np
import pandas as pd
import multiprocessing
import torch.multiprocessing as mp
from tqdm import tqdm
from datetime import datetime
from colorlog import ColoredFormatter
from typing import List, Dict, Any, Tuple, Union, Literal



from produce_adjacent_matrix import build_edge_list
from train_test_custom import parallel_train_test_one_target
from log_config import initialize_logging


def randomize_run_name(): return f"{random.choice(royals_name)}_{random.choice(royals_name)}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# -----------------------------------------------------------------------------------------------
# Log Configuration
# -----------------------------------------------------------------------------------------------
initialize_logging()




# -----------------------------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # -----------------------------------------------------------------------------------------------
    # General Configuration
    # -----------------------------------------------------------------------------------------------
    from env_config import *

    if args.parallel:
        multiprocessing.set_start_method("spawn", force=True)
        mp.set_start_method("spawn", force=True)



    run_name = 'baseline_MCRGCN' if args.run_mode != "test" else randomize_run_name()
    logging.info(f"Starting MCRGCN evaluation on {args.run_mode} mode, storage mode {args.storage_mode}")
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



    def find_run(collection, run_id: str, target_id: str): return collection.find_one({'run_id': run_id, 'target_id': target_id})
    # -----------------------------------------------------------------------------------------------
    # MLFlow
    # -----------------------------------------------------------------------------------------------
    mlflow.set_tracking_uri(uri="http://127.0.0.1:6969")
    mlflow.set_experiment(experiment_name)




    # -----------------------------------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------------------------------
    miRNA = pd.read_parquet(f"{DATA_PATH}/miRNA.parquet", storage_options=storage_options)
    mRNA = pd.read_parquet(f"{DATA_PATH}/mRNA.parquet", storage_options=storage_options)
    target_folders = [f's3://{a}' for a in s3.ls(TARG_PATH)] if s3 is not None else os.listdir(TARG_PATH)



    # -----------------------------------------------------------------------------------------------
    # Sample Sim.matrix
    # -----------------------------------------------------------------------------------------------
    mRNA_sim = build_edge_list(mRNA.T, 0.35)
    miRNA_sim = build_edge_list(miRNA.T, 0.42)
    mRNA_sim = mRNA_sim[['id_x', 'id_y']].values
    miRNA_sim = miRNA_sim[['id_x', 'id_y']].values
        



    # -----------------------------------------------------------------------------------------------
    # Run ID Retrieval
    # -----------------------------------------------------------------------------------------------
    all_runs = mlflow.search_runs()[['run_id', 'tags.mlflow.runName']]
    possible_run_ids = all_runs[all_runs['tags.mlflow.runName'] == run_name]
    run_id = possible_run_ids['run_id'].values[0] if len(possible_run_ids) > 0 else None



    # -----------------------------------------------------------------------------------------------
    # Parrallel Processing
    # -----------------------------------------------------------------------------------------------
    if args.parallel:
        result_queue = mp.Queue()
        processes = []
    else:
        result_queue = []


    # -----------------------------------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------------------------------
    for fol_id, target_folder in enumerate(target_folders):
        # Retrieve test data
        target_id = str(target_folder.split('/')[-1]).split('.')[0]
        if run_id is not None:
            if find_run(collection, run_id, target_id) is not None:
                logging.info(f"Run {run_id} on dataset {target_id} already exists. Skipping")
                continue
        test_data = pd.read_parquet(target_folder, storage_options=storage_options)
        armed_gpu = fol_id % torch.cuda.device_count()


        # Evaluate
        if args.parallel:
            process = mp.Process(
                target = parallel_train_test_one_target,
                args = (
                    [mRNA.to_dict(orient='index'), miRNA.to_dict(orient='index')],
                    [mRNA_sim, miRNA_sim],
                    test_data.to_dict(orient='index'),
                    armed_gpu,
                    target_id,
                    result_queue,
                )
            )    

            logging.info(f"Starting process for target {target_id} on device {armed_gpu}")
            process.start()
            processes.append(process)
        else:
            sequential_result = parallel_train_test_one_target(
                omic_layers=[mRNA.to_dict(orient='index'), miRNA.to_dict(orient='index')],
                omic_sims=[mRNA_sim, miRNA_sim],
                test_data=test_data.to_dict(orient='index'),
                armed_gpu=armed_gpu,
                target_id=target_id,
                result_queue=None,
            )
            result_queue.append(sequential_result)
            logging.info(f"Finished evaluation for target {target_id}")
    

    # -----------------------------------------------------------------------------------------------
    # Sync processes
    # -----------------------------------------------------------------------------------------------
    if args.parallel:
        for process in tqdm(processes, desc="Waiting for processes to finish"):
            process.join()
        parallel_results = []
        while not result_queue.empty():
            parallel_results.append(result_queue.get())
        result_queue = parallel_results



    # -----------------------------------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------------------------------
    with mlflow.start_run(run_id = run_id) if run_id is not None else mlflow.start_run(run_name=run_name):
        logging.info(f"Initializing run {run_name}")
        if run_id is None: 
            run_id = mlflow.active_run().info.run_id
            with s3.open(f"{RESULT_PRE_PATH}/{run_name}/run_id.txt", 'w') as f:
                f.write(run_id)

            mlflow.log_param("Number of omics layers", 2)
            mlflow.log_param("Omics layers feature size", [mRNA.shape[0], miRNA.shape[0]])
            mlflow.log_param("Sample size", miRNA.shape[1])
        logging.info(f"Run {run_id} initialized")

        for result in result_queue:
            target_id = result['id']
            result_pack = pd.DataFrame.from_dict(result['data'], orient='index')


            data_pack = {
                'run_id': run_id,
                'run_name': run_name,
                'target_id': target_id,
                'summary': {}
            }
            
            for metric in result_pack.columns:
                if str(metric).isupper():
                    # Assume all metrics are upper case-noted columns
                    data_pack['summary'][f'Mean {metric}'] = np.mean(result_pack[metric].values)
                    data_pack['summary'][f'Median {metric}'] = np.median(result_pack[metric].values)
                    data_pack['summary'][f'Std {metric}'] = np.std(result_pack[metric].values)
                    data_pack['summary'][f'Max {metric}'] = np.max(result_pack[metric].values)
                    data_pack['summary'][f'Min {metric}'] = np.min(result_pack[metric].values)

            logging.info(f"========================================================\n\n\n\n")
            logging.info(f"Summary for target {target_id}")
            for key in data_pack['summary'].keys():
                logging.info(f"{key}: {data_pack['summary'][key]}")

                if 'Mean AUROC' in key: 
                    for method in classification_methods_list:
                        mlflow.log_metric(f'{target_id} {method} Mean AUC', data_pack['summary'][key])
                if 'Mean MCC' in key: mlflow.log_metric(f'{target_id} {key}', data_pack['summary'][key])


            # Save to MongoDB
            collection.insert_one(data_pack)
    



    



# Content and code by bu1th4nh. Written with dedication in the University of Central Florida, EPCOT and the Magic Kingdom.
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