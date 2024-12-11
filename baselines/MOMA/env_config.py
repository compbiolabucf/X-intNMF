# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: env_config.py
# Date: 2024/12/10 12:48:18
# Description: 
# 
# (c) 2024 bu1th4nh. All rights reserved. 
# Written with dedication in the University of Central Florida, EPCOT and the Magic Kingdom.
# -----------------------------------------------------------------------------------------------




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
parser.add_argument("--gpu", type=int, required=False, default=0)
parser.add_argument("--parallel", type=bool, required=False, default=False)
args = parser.parse_args()



# -----------------------------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------------------------
if args.storage_mode == "local":
    DATA_PATH = '/home/ti514716/Datasets'
    RESULT_PRE_PATH = '/home/ti514716/Results/SimilarSampleCrossOmicNMF/'
    storage_options = None
    s3 = None
elif args.storage_mode == "s3":
    DATA_PATH = 's3://datasets'
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
    base_path = f'{DATA_PATH}/LungCancer'
    DATA_PATH = f'{base_path}/processed'
    TARG_PATH = f'{base_path}/clinical_testdata'
    RESULT_PRE_PATH += 'luad'
elif args.run_mode == "ov":
    experiment_name = 'SimilarSampleCrossOmicNMFv3_OV'
    base_path = f'{DATA_PATH}/OvarianCancer'
    DATA_PATH = f'{base_path}/processed'
    TARG_PATH = f'{base_path}/clinical_testdata'
    RESULT_PRE_PATH += 'ov'
elif args.run_mode == "brca":
    experiment_name = 'SimilarSampleCrossOmicNMFv3'
    base_path = f'{DATA_PATH}/BreastCancer'
    DATA_PATH = f'{base_path}/processed_crossOmics'
    TARG_PATH = f'{base_path}/clinical_testdata'
    RESULT_PRE_PATH += 'brca'
elif args.run_mode == "test":
    experiment_name = 'test_experiment'
    base_path = f'{DATA_PATH}/BreastCancer'
    DATA_PATH = f'{base_path}/processed_crossOmics'
    TARG_PATH = f'{base_path}/clinical_testdata'
    RESULT_PRE_PATH += 'brca'
else: raise ValueError("Invalid run mode")

