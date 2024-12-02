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
import s3fs

royals_name = ['snowwhite', 'cinderella', 'aurora', 'ariel', 'belle', 'jasmine', 'pocahontas', 'mulan', 'tiana', 'rapunzel', 'merida', 'moana', 'raya', 'anna', 'elsa', 'elena']
classification_methods_list = ["Logistic Regression", "Random Forest"]
logfile_name = "model.log"
pickup_leftoff_mode = True


# experiment_name = "test_experiment"
# storage_options = None
# DATA_PATH = '/home/ti514716/Datasets/LungCancer/processed_micro'
# RESULT_PRE_PATH = '/home/ti514716/Results/Test'
# s3 = None



# experiment_name = 'SimilarSampleCrossOmicNMFv3_LUAD'
# storage_options = None
# DATA_PATH = '/home/bu1th4nh/Datasets/LungCancer/processed'
# RESULT_PRE_PATH = '/home/bu1th4nh/Results/SimilarSampleCrossOmicNMF/luad'
# s3 = None



experiment_name = 'SimilarSampleCrossOmicNMFv3_LUAD'
storage_options = {
    'key': 'bu1th4nh',
    'secret': 'ariel.anna.elsa',
    'endpoint_url': 'http://localhost:9000',
}
DATA_PATH = 's3://datasets/LungCancer/processed'
RESULT_PRE_PATH = 's3://results/SimilarSampleCrossOmicNMF/luad'
s3 = s3fs.S3FileSystem(
    key=storage_options['key'],
    secret=storage_options['secret'],
    endpoint_url=storage_options['endpoint_url'],
    use_ssl=False,
)
