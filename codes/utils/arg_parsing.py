# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: arg_parsing.py
# Date: 2025/03/24 16:59:57
# Description: Helper functions for parsing arguments in the command line.
# 
# This program/software is licensed under MIT License
# Copyright (c) 2025 Tien-Thanh Bui (bu1th4nh) / UCF Computational Biology Lab.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
# This software is written with dedication in the University of Central Florida, EPCOT and the Magic Kingdom.
# -----------------------------------------------------------------------------------------------


import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Union, Literal


# -----------------------------------------------------------------------------------------------
# Config & Args
# -----------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="X-intNMF - A graph-regularized NMF framework to integrate the multi-omics data with cross&inter-omics interactions.",
    epilog="(c) 2025 bu1th4nh / UCF Computational Biology Lab / University of Central Florida. Written with dedication in the University of Central Florida, EPCOT and the Magic Kingdom.",
    # usage="""
    #     python X-intNMF-run.py 
    #     --omics_input <omics_input> 
    #     --interaction_input <interaction_input> 
    #     --output_dir <output_dir> 
    #     --num_components <num_components> 
    #     --num_iterations <num_iterations> 
    # """
)

############ 
# Data
############
parser.add_argument(
    "--omics_input",
    metavar="/path/to/omics_data",
    nargs="+",
    type=str,
    required=True,
    help="List of omics input files path seperated by whitespace character. Each omics data file must have the file extension .csv, .parquet, .tsv, .txt or .xlsx. If the path has whitespaces, please enclose the path in double quotes. For example: --omics_input \"path/to/omics1.csv\" \"path/to/omics2.parquet\".",
) 
parser.add_argument(
    "--interaction_input",
    metavar="/path/to/interaction_data",
    nargs="+",
    type=str,
    required=True,
    help="List of interaction input files path seperated by whitespace character. Each interaction data file must have the file extension .csv, .parquet, .tsv, .txt or .xlsx and the name must have the form interaction_<omics1 filename>_<omics2 filename>.<filetype>. For example: \"interaction_omics1_omics2.parquet\", the program will interpret the row as omics1, column as omics2 and the value will be the interaction score between omics1 and omics2.",
)


############
# Output
############
parser.add_argument(
    "--output_format",
    choices=["csv", "parquet", "tsv", "xlsx"],
    type=str,
    required=False,
    default="parquet",
    help="Output format of the result files. Default is parquet for optimal performance. Other options are csv, tsv and xlsx.",
)
parser.add_argument(
    "--output_dir",
    metavar="/path/to/desired/output/directory",
    type=str,
    required=False,
    default="./output",
    help="Output directory to save the results. Default is ./output.",
)


############
# Parameters
############
parser.add_argument(
    "--num_components",
    metavar="k",
    type=int,
    required=False,
    default=25,
    help="Number of latent components to use for NMF. Default is 25.",
)
parser.add_argument(
    "--graph_regularization",
    metavar="alpha",
    type=float,
    required=False,
    default=1,
    help="Graph regularization parameter. Default is 1.",
)
parser.add_argument(
    "--omics_regularization",
    metavar="beta",
    type=float,
    required=False,
    default=0.1,
    help="Omics regularization parameter. Default is 0.1.",
)
parser.add_argument(
    "--sample_regularization",
    metavar="gamma",
    type=float,
    required=False,
    default='-1',
    help="Sample regularization parameter. Default is auto-generated by Lasso.",
)


############
# Settings
############
parser.add_argument(
    "--max_iter",
    metavar="max_iter",
    type=int,
    required=False,
    default=5000,
    help="Maximum iterations. Default is 5000.",
)
parser.add_argument(
    "--tol",
    metavar="eps",
    type=float,
    required=False,
    default=1e-4,
    help="Tolerance for stopping criteria. Default is 1e-4.",
)
parser.add_argument(
    "--gpu",
    metavar="gpu",
    type=int,
    required=False,
    default=-1,
    help="GPU device ID. Default is -1, which means CPU. If you want to use GPU, please set the device number, which is the integer from 0 to (number of GPU - 1). For example: --gpu 0.",
)
parser.add_argument(
    "--backend",
    choices=["numpy", "pytorch"],
    required=False,
    default="numpy",
    help="Backend to use for computation. Default is numpy. Other option is pytorch. However, if you set the gpu device, the backend will be automatically set to pytorch.",
)
parser.add_argument(
    "--log_file",
    metavar="/path/to/log_file",
    type=str,
    required=False,
    default="x_intnmf.log",
    help="Path to log file. Default is \"x_intnmf.log\", which means no log file. If you want to save the log to a file, please set the path to the log file.",
)
parser.add_argument(
    "--mlflow_uri",
    metavar="<mlflow_host>:<port>",
    type=str,
    required=False,
    default="",
    help="MLflow server URI for convergence log. Default is empty string, which means no MLflow server. If you want to use MLflow server, please set the URI in the format of <mlflow_host>:<port>. For example: --mlflow_uri \"localhost:5000\".",
)
parser.add_argument(
    "--mlflow_experiment_name",
    metavar="experiment_name",
    type=str,
    required=False,
    default="X-intNMF",
    help="MLflow experiment name. Default is X-intNMF. Only used when --mlflow_uri is set.",
)
args = parser.parse_args()



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