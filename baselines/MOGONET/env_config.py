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
parser.add_argument("--storage_mode", type=str, required=True)
parser.add_argument("--omics_mode", type=str, required=True)
parser.add_argument("--disease", type=str, required=True)
parser.add_argument("--gpu", type=int, required=False, default=0)
parser.add_argument("--parallel", type=bool, required=False, default=False)
args = parser.parse_args()



# -----------------------------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------------------------
# Storage
if args.storage_mode == "local":
    base_data_path = '/home/ti514716/Datasets'
    base_result_path = '/home/ti514716/Results'
    storage_options = None
    s3 = None
elif args.storage_mode == "s3":
    base_data_path = 's3://datasets'
    base_result_path = 's3://results'
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


# Omics
if args.omics_mode == "3omics":
    mongo_db_name           = 'SimilarSampleCrossOmicNMF_3Omics'
    base_result_path        = f'{base_result_path}/SimilarSampleCrossOmicNMF_3Omics'
    omic_folder             = 'processed_3_omics_mRNA_miRNA_methDNA'
    cls_target_folder       = 'clinical_testdata_3_omics_mRNA_miRNA_methDNA'
    surv_target_folder      = 'survival_testdata_3_omics_mRNA_miRNA_methDNA'
    experiment_addon_ext    = '_3Omics'
elif args.omics_mode == "2omics":
    mongo_db_name           = 'SimilarSampleCrossOmicNMF'
    base_result_path        = f'{base_result_path}/SimilarSampleCrossOmicNMF'
    omic_folder             = 'processed_2_omics_mRNA_miRNA'
    cls_target_folder       = 'clinical_testdata_2_omics_mRNA_miRNA'
    surv_target_folder      = 'survival_testdata_2_omics_mRNA_miRNA'
    experiment_addon_ext    = ''


# Disease
if args.disease == "brca":
    dataset_id              = 'BRCA'
    mongo_collection        = 'BRCA'
    disease_data_folder     = 'BreastCancer'
    disease_result_folder   = 'brca'
    experiment_name         = f'SimilarSampleCrossOmicNMFv3_BRCA{experiment_addon_ext}'
elif args.disease == "luad":
    dataset_id              = 'LUAD'
    mongo_collection        = 'LUAD'
    disease_data_folder     = 'LungCancer'
    disease_result_folder   = 'luad'
    experiment_name         = f'SimilarSampleCrossOmicNMFv3_LUAD{experiment_addon_ext}'
elif args.disease == "ov":
    dataset_id              = 'OV'
    mongo_collection        = 'OV'
    disease_data_folder     = 'OvarianCancer'
    disease_result_folder   = 'ov'
    experiment_name         = f'SimilarSampleCrossOmicNMFv3_OV{experiment_addon_ext}'
elif args.disease == "test":
    dataset_id              = 'test'
    mongo_collection        = 'TEST'
    disease_data_folder     = 'BreastCancer'
    disease_result_folder   = 'test'
    experiment_name         = 'test_experiment'






# Aggregate
TARG_PATH = f'{base_data_path}/{disease_data_folder}/{cls_target_folder}'
RUN_CFG_PATH = f'{base_result_path}/{disease_result_folder}'
DATA_PATH = f'{base_data_path}/{disease_data_folder}/{omic_folder}'
RESULT_PRE_PATH = f'{base_result_path}/{disease_result_folder}'


