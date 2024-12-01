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

# DATA_PATH = '/home/ti514716/Datasets/BreastCancer/processed_crossOmics_micro'
DATA_PATH = '/home/ti514716/Datasets/BreastCancer/processed_crossOmics'
result_pre_path = '/home/ti514716/Projects/SimilarSampleCrossOmicNMF/results/v3_brca'

royals_name = ['snowwhite', 'cinderella', 'aurora', 'ariel', 'belle', 'jasmine', 'pocahontas', 'mulan', 'tiana', 'rapunzel', 'merida', 'moana', 'raya', 'anna', 'elsa', 'elena']
logfile_name = "model.log"
experiment_name = "SimilarSampleCrossOmicNMFv3"


pickup_leftoff_mode = True
classification_methods_list = ["Logistic Regression", "Random Forest"]