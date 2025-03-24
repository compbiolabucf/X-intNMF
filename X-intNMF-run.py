# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: X-intNMF-run.py
# Date: 2025/03/17 23:15:12
# Description: 
# 
# (c) 2025 bu1th4nh / UCF Computational Biology Lab. All rights reserved. 
# Written with dedication in the University of Central Florida, EPCOT and the Magic Kingdom.
# -----------------------------------------------------------------------------------------------


import os
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Union, Literal


from codes.utils.files import autoread_file, strip_filename, autosave_file
from codes.utils.log_config import initialize_logging
from codes.utils.arg_parsing import *
from codes.crossOmicNMF import XIntNMF



initialize_logging(log_filename=args.log_file)
# -----------------------------------------------------------------------------------------------
# Input Processing
# -----------------------------------------------------------------------------------------------
omic_paths = args.omics_input
interaction_paths = args.interaction_input


# Dataset
omics_data = [autoread_file(path) for path in omic_paths]
omics_filenames = { strip_filename(path): idx for idx, path in enumerate(omic_paths) }
omics_id_array = [strip_filename(path) for path in omic_paths]
omics_features = [data.index.tolist() for data in omics_data]
samples = omics_data[0].columns.tolist()


# Interaction
interaction_data = {}
for path in interaction_paths:
    # Extract omics names from the filename
    filename = strip_filename(path)
    omics1, omics2 = filename.split("_")[1:3]

    # Sanitize omics names, i.e. check if they are in the omics_filenames dict
    if omics1 not in omics_filenames or omics2 not in omics_filenames:

        missing = []
        if omics1 not in omics_filenames: missing.append(omics1)
        if omics2 not in omics_filenames: missing.append(omics2)
        missing = " and ".join(missing) if len(missing) > 1 else missing[0]
    
        raise ValueError(f"Invalid interaction filename: {filename}. The program inteprets the filename as interaction between {omics1} and {omics2}, which are filenames excluding the file extension of the omics data files. However, {missing} not found in the omics data files. Please check the filename and the omics data files.")
    
    # Read the interaction data
    o1_index = omics_filenames[omics1]
    o2_index = omics_filenames[omics2]
    interaction_data[(o1_index, o2_index)] = autoread_file(path).to_numpy(np.float64, True, na_value=0.0)



# -----------------------------------------------------------------------------------------------
# Parameters & Settings
# -----------------------------------------------------------------------------------------------
k = args.num_components
alpha = args.graph_regularization
betas = args.omics_regularization
gammas = args.sample_regularization

max_iter = args.num_iterations
tol = args.tolerance
gpu = args.gpu
backend = args.backend if gpu != -1 else "pytorch"
mlflow = args.mlflow_uri
mlflow_expr_name = args.mlflow_experiment_name

out_format = args.output_format
out_dir = args.output_dir
# -----------------------------------------------------------------------------------------------
# MLFlow
# -----------------------------------------------------------------------------------------------
if len(mlflow) > 0:
    import mlflow
    mlflow.set_tracking_uri(mlflow)
    mlflow.set_experiment(mlflow_expr_name)
    run_name = f"X-intNMF-components={k}-alpha={alpha}-beta={betas}-gamma={gammas}-backend={backend}"
    mlflow.start_run(run_name=run_name)



# -----------------------------------------------------------------------------------------------
# X-intNMF
# -----------------------------------------------------------------------------------------------
model = XIntNMF(
    omics_layers = [data.to_numpy(np.float64, True, na_value=0.0) for data in omics_data],
    cross_omics_interaction = interaction_data,
    k = k,
    alpha = alpha,
    betas = betas,
    gammas = 1 if gammas == -1 else gammas,
    max_iter = max_iter,
    tol = tol,
    verbose = True,
    backend = backend,
    gpu = None if gpu == -1 else gpu,
    mlflow_enable = True if mlflow != "" else False,
)


# Solve
Ws, H = model.solve(run_mode='full', use_cross_validation=(gammas == -1))

# -----------------------------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------------------------
# Check output directory
if out_dir.endswith("/"): out_dir = out_dir[:-1]
if not os.path.exists(out_dir): 
    logging.info(f"Output directory {out_dir} does not exist. Creating...")
    os.makedirs(out_dir, exist_ok=True)

# Init columns and paths
latent_columns = [f"Latent_{i:03}" for i in range(H.shape[0])]

# Save Ws
for W, filename, feature_list in zip(Ws, omics_id_array, omics_features):
    df_W = pd.DataFrame(W, index=feature_list, columns=latent_columns)
    path = f"{out_dir}/{filename}_factor.{out_format}"
    autosave_file(df_W, out_format, path)
    logging.info(f"Saved {filename} output factor to: {path}")
    
# Save H
df_H = pd.DataFrame(H.T, index=samples, columns=latent_columns)
path = f"{out_dir}/sample_factor.{out_format}"
autosave_file(df_H, out_format, path)
logging.info(f"Saved sample output factor to: {path}")







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
