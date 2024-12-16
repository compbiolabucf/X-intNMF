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
parser.add_argument("--gpu", type=int, required=False, default=0)
parser.add_argument("--parallel", type=bool, required=False, default=False)
args = parser.parse_args()



# -----------------------------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------------------------
if args.storage_mode == "local":
    base_data_path = '/home/ti514716/Datasets'
    base_result_path = '/home/ti514716/Results/SimilarSampleCrossOmicNMF'
    storage_options = None
    s3 = None
elif args.storage_mode == "s3":
    base_data_path = 's3://datasets'
    base_result_path = 's3://results/SimilarSampleCrossOmicNMF'
    storage_options = {
        'key': 'bu1th4nh',
        'secret': 'ariel.anna.elsa',
        'endpoint_url': 'http://localhost:19000',
    }
    s3 = s3fs.S3FileSystem(
        key=storage_options['key'],
        secret=storage_options['secret'],
        endpoint_url=storage_options['endpoint_url'],
        use_ssl=False,
    )
else: raise ValueError("Invalid storage mode")
    


if args.run_mode == "luad":
    experiment_name  = 'SimilarSampleCrossOmicNMFv3_LUAD'
    mongo_collection = 'LUAD'
    DATA_PATH        = f'{base_data_path}/LungCancer/processed'
    TARG_PATH        = f'{base_data_path}/LungCancer/clinical_testdata'
    RESULT_PRE_PATH  = f'{base_result_path}/luad'

elif args.run_mode == "ov":
    experiment_name  = 'SimilarSampleCrossOmicNMFv3_OV'
    mongo_collection = 'OV'
    DATA_PATH        = f'{base_data_path}/OvarianCancer/processed'
    TARG_PATH        = f'{base_data_path}/OvarianCancer/clinical_testdata'
    RESULT_PRE_PATH  = f'{base_result_path}/ov'

elif args.run_mode == "brca":
    experiment_name  = 'SimilarSampleCrossOmicNMFv3'
    mongo_collection = 'BRCA'
    DATA_PATH        = f'{base_data_path}/BreastCancer/processed_crossOmics'
    TARG_PATH        = f'{base_data_path}/BreastCancer/clinical_testdata'
    RESULT_PRE_PATH  = f'{base_result_path}/brca'

elif args.run_mode == "hparams_opts_luad":
    experiment_name  = 'SimilarSampleCrossOmicNMFv3_LUAD'
    mongo_collection = 'HPARAMS_OPTS'
    dataset_id       = 'LUAD'
    DATA_PATH        = f'{base_data_path}/LungCancer/processed'
    TARG_PATH        = f'{base_data_path}/LungCancer/clinical_testdata'
    RUN_CFG_PATH     = f'{base_result_path}/luad'

elif args.run_mode == "hparams_opts_ov":
    experiment_name  = 'SimilarSampleCrossOmicNMFv3_OV'
    mongo_collection = 'HPARAMS_OPTS'
    dataset_id       = 'OV'
    DATA_PATH        = f'{base_data_path}/OvarianCancer/processed'
    TARG_PATH        = f'{base_data_path}/OvarianCancer/clinical_testdata'
    RUN_CFG_PATH     = f'{base_result_path}/ov'
    
elif args.run_mode == "hparams_opts_brca":
    experiment_name  = 'SimilarSampleCrossOmicNMFv3'
    mongo_collection = 'HPARAMS_OPTS'
    dataset_id       = 'BRCA'
    DATA_PATH        = f'{base_data_path}/BreastCancer/processed_crossOmics'
    TARG_PATH        = f'{base_data_path}/BreastCancer/clinical_testdata'
    RUN_CFG_PATH     = f'{base_result_path}/brca'



    
else: raise ValueError("Invalid run mode")


if cp.cuda.is_available():
    gpu = np.clip(args.gpu, 0, cp.cuda.runtime.getDeviceCount()-1)
    cp.cuda.runtime.setDevice(gpu)