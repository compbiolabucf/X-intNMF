# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: main_moma_custom.py
# Date: 2024/11/21 14:13:34
# Description: 
# 
# (c) 2024 bu1th4nh. All rights reserved. 
# Written with dedication in the University of Central Florida, EPCOT and the Magic Kingdom.
# -----------------------------------------------------------------------------------------------


import sys
import torch
import mlflow
import logging
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
from colorlog import ColoredFormatter
from typing import List, Dict, Any, Tuple, Union, Literal


from train_test import custom___train_test, parallel_train_test

DATA_PATH = "/home/bu1th4nh/Datasets/BreastCancer/processed_crossOmics"
# -----------------------------------------------------------------------------------------------
# Log Configuration
# -----------------------------------------------------------------------------------------------
logging.root.handlers = [];

# Console handler
handler_sh = logging.StreamHandler(sys.stdout)
handler_sh.setFormatter(
    ColoredFormatter(
        "%(cyan)s%(asctime)s.%(msecs)03d %(log_color)s[%(levelname)s]%(reset)s %(light_white)s%(message)s%(reset)s %(blue)s(%(filename)s:%(lineno)d)",
        datefmt  = '%Y/%m/%d %H:%M:%S',
        log_colors={
            'DEBUG': 'white',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
)
logging.basicConfig(
    level    = logging.INFO,
    handlers = [handler_sh]
)



# -----------------------------------------------------------------------------------------------
# Patch NumPy
# -----------------------------------------------------------------------------------------------
import numpy
def patch_asscalar(a):
    return a.item()
setattr(numpy, "asscalar", patch_asscalar)


# -----------------------------------------------------------------------------------------------
# MLFlow Configuration
# -----------------------------------------------------------------------------------------------
mlflow.set_tracking_uri("http://localhost:6969")
mlflow.set_experiment("SimilarSampleCrossOmicNMFv3")
# mlflow.set_experiment("test_experiment")




# -----------------------------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------------------------
testdata = pd.read_parquet(f'{DATA_PATH}/testdata_classification.parquet')
mRNA = pd.read_parquet(f'{DATA_PATH}/mRNA.parquet')
miRNA = pd.read_parquet(f'{DATA_PATH}/miRNA.parquet')
label_data = pd.read_parquet(f'{DATA_PATH}/clinical.parquet')
for label in label_data.columns:
    label_data[label] = label_data[label].apply(lambda x: 1 if x == 'Positive' else 0)
    
# -----------------------------------------------------------------------------------------------
# Multiprocessing
# -----------------------------------------------------------------------------------------------
result_queue = multiprocessing.Queue()
processes = []


# -----------------------------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------------------------
logging.info("Starting MOMA evaluation") 
# Please note the methods is just placeholder and for easier logging and comparison to MLFlow, 
# MOMA directly predicts the label, not through classifiers
methods_list = ["Logistic Regression", "Random Forest"] 
with mlflow.start_run(run_name="baseline_MOMA"):


    # MLFlow logging
    # mlflow.log_param("alpha", alpha)
    # mlflow.log_param("betas", betas)
    # mlflow.log_param("gammas", gammas)
    # mlflow.log_param("max_iter", self.max_iter)
    # mlflow.log_param("tol", self.tol)
    # mlflow.log_param("verbose", self.verbose)

    mlflow.log_param("Number of omics layers", 2)
    mlflow.log_param("Sample size", mRNA.shape[1])
    mlflow.log_param("Omics layers feature size", [mRNA.shape[0], miRNA.shape[0]])
    # mlflow.log_param("Latent size", )

    # Prepping the data and result
    logging.info("Starting evaluation")
    auc_columns = []
    for label in label_data.columns:
        for method in methods_list:
            auc_columns.append(f"{label}_{method}_AUC")
    AUC_result = pd.DataFrame(
        index = testdata.index,
        columns = auc_columns
    )

    # Assume label is 0-1
    # and there are 4 labels in the label_data, = nr. of GPU
    for dev_id, label in enumerate(label_data.columns):
        logging.info(f"Building process for label {label} on device {dev_id % torch.cuda.device_count()}")
        process = multiprocessing.Process(
            target = parallel_train_test,
            args = (
                f"cuda:{int(dev_id % torch.cuda.device_count())}",
                label_data.copy(deep=True),
                testdata.copy(deep=True),
                label,
                pd.Series(methods_list).copy(deep=True).tolist(),
                mRNA.copy(deep=True),
                miRNA.copy(deep=True),
                result_queue
            )
        )

        logging.info(f"Starting process for label {label} on device {dev_id % torch.cuda.device_count()}")
        process.start()
        processes.append(process)



    # Wait for all processes to finish
    # logging.info("Waiting for processes to finish")
    for process in tqdm(processes, desc="Waiting for processes to finish"):
        process.join()


    # Collect results
    logging.info("Collecting results")
    while not result_queue.empty():
        result = result_queue.get()
        label = result["label"]
        auc = result["auc"]

        for method in methods_list:
            for test_id in auc.keys():
                AUC_result.loc[test_id, f"{label}_{method}_AUC"] = auc[test_id]

        




    # Calculate, log statistics
    logging.info("Eval completed. Calculating statistics")
    AUC_result.to_parquet(f"MOMA_AUC_result.parquet")
    for label in label_data.columns:
        for cls_method in methods_list:
            auc_values = AUC_result[f"{label}_{cls_method}_AUC"].values
            avg_auc = np.nanmean(auc_values)
            std_auc = np.nanstd(auc_values)
            max_auc = np.nanmax(auc_values)
            min_auc = np.nanmin(auc_values)
            med_auc = np.nanmedian(auc_values)
            
            
            # Logging
            logging.info(f"{label} - {method} - Mean AUC: {avg_auc:.05f}")
            logging.info(f"{label} - {method} - Median AUC: {med_auc:.05f}")
            logging.info(f"{label} - {method} - Std AUC: {std_auc:.05f}")
            logging.info(f"{label} - {method} - Max AUC: {max_auc:.05f}")
            logging.info(f"{label} - {method} - Min AUC: {min_auc:.05f}")
            print()

            # MLFlow
            mlflow.log_metric(f"{label} {cls_method} Mean AUC", avg_auc)


    # Save the result



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