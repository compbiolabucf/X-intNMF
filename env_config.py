# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành (@bu1th4nh)
# Title: env_config.py
# Date: 2024/10/19 16:23:28
# Description: 
# 
# (c) bu1th4nh. All rights reserved
# -----------------------------------------------------------------------------------------------


import cupy as cp
import numpy as np
import pandas as pd
import argparse
import s3fs


# -----------------------------------------------------------------------------------------------
# General
# -----------------------------------------------------------------------------------------------
royals_name = ['snowwhite', 'cinderella', 'aurora', 'ariel', 'belle', 'jasmine', 'pocahontas', 'mulan', 'tiana', 'rapunzel', 'merida', 'moana', 'raya', 'anna', 'elsa', 'elena']
classification_methods_list = ["Logistic Regression", "Random Forest"]
logfile_name = "model.log"
pickup_leftoff_mode = True


# -----------------------------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--run_mode", type=str, required=True)
parser.add_argument("--storage_mode", type=str, required=True)
parser.add_argument("--gpu", type=int, required=True)
args = parser.parse_args()



# -----------------------------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------------------------
if args.storage_mode == "local":
    DATA_PATH = '/home/ti514716/Datasets/'
    RESULT_PRE_PATH = '/home/ti514716/Results/SimilarSampleCrossOmicNMF/'
    storage_options = None
    s3 = None
elif args.storage_mode == "s3":
    DATA_PATH = 's3://datasets/'
    RESULT_PRE_PATH = 's3://results/SimilarSampleCrossOmicNMF/'
    storage_options = {
        'key': 'bu1th4nh',
        'secret': 'ariel.anna.elsa',
        'endpoint_url': 'http://localhost:9000',
    }
    s3 = s3fs.S3FileSystem(
        key=storage_options['key'],
        secret=storage_options['secret'],
        endpoint_url=storage_options['endpoint_url'],
        use_ssl=False,
    )
else: raise ValueError("Invalid storage mode")
    


if args.run_mode == "luad":
    experiment_name = 'SimilarSampleCrossOmicNMFv3_LUAD'
    DATA_PATH += 'LungCancer/processed'
    RESULT_PRE_PATH += 'luad'
elif args.run_mode == "ov":
    experiment_name = 'SimilarSampleCrossOmicNMFv3_OV'
    DATA_PATH += 'OvarianCancer/processed'
    RESULT_PRE_PATH += 'ov'
elif args.run_mode == "brca":
    experiment_name = 'SimilarSampleCrossOmicNMFv3'
    DATA_PATH += 'BreastCancer/processed'
    RESULT_PRE_PATH += 'brca'
else: raise ValueError("Invalid run mode")



gpu = np.clip(args.gpu, 0, cp.cuda.runtime.getDeviceCount()-1)
cp.cuda.runtime.setDevice(gpu)