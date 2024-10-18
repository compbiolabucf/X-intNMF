# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành (@bu1th4nh)
# Title: config.py
# Date: 2024/10/07 23:06:39
# Description: 
# 
# (c) bu1th4nh. All rights reserved
# -----------------------------------------------------------------------------------------------



ORGL_PATH = '/home/ti514716/Datasets/BreastCancer/processed_crossOmics'
RESULT_PATH = '/home/ti514716/Projects/SimilarSampleCrossOmicNMF/results'
METHODS = ["AdaBoost", "Logistic Regression", "Random Forest", "SVM"]
DISABLE_MLFLOW = False
DISABLE_FILE = False
TRAIN_TIME = 1000


run_presets = [
    # Not fixed gamma
    # Alpha = 2
    {
        'run_name'  : 'ariel-elsa-aurora-20241002-15.12.43',
        'run_id'    : 'd4cf242b1a3540d5b2cb91dc52dbd991',
    },
    # Alpha = 1
    {
        'run_name'  : 'ariel-moana-mulan-20241002-23.32.36',
        'run_id'    : '1e2f0bbe6ada401aae9d027bcd0fa8b5',
    },
    # Fixed gamma
    {
        'run_name'  : 'merida-mulan-anna-20241003-08.56.41',
        'run_id'    : 'abc7d2cfd6d04ab7b48de90d24e8cfc4',
    },
    # Baseline: NMF Only
    {
        'run_name'  : 'rapunzel-rapunzel-ariel-20241006-10.23.03',
        'run_id'    : '7f933693ee25409c8fdb90b04a5a26b8',
    },
    # Baseline: Raw
    {
        'run_name'  : 'baseline_rawdata',
        'run_id'    : '62565423ab374538bddded038085c625',
    },
]
