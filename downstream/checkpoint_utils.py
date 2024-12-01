# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: checkpoint_utils.py
# Date: 2024/12/01 16:42:08
# Description: Utility functions for checkpointing and resuming DR
# 
# (c) 2024 bu1th4nh. All rights reserved. 
# Written with dedication in the University of Central Florida, EPCOT and the Magic Kingdom.
# -----------------------------------------------------------------------------------------------

import os
import logging
import cupy as cp
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Union, Literal




class IterativeCheckpointing:
    def __init__(self, sample_list, omics_features, prefix_path):
        self.sample_list = sample_list
        self.omics_features = omics_features
        self.prefix_path = prefix_path
        self.checkpoint_path = f"{prefix_path}/checkpoints"

        os.makedirs(self.checkpoint_path, exist_ok=True)
        logging.fatal(f"Checkpointing to: {self.checkpoint_path}")



    def save(
        self, 
        Ws: Union[List[np.ndarray], List[cp.ndarray]],
        H: Union[np.ndarray, cp.ndarray],
        step: Union[int, None],
    ):
        # CuPy Sanitize
        if isinstance(H, cp.ndarray): 
            H = H.get()
            Ws = [W.get() for W in Ws]


        # Init columns and paths
        latent_columns = [f"Latent_{i:03}" for i in range(H.shape[0])]
        actual_checkpoint_path = f"{self.prefix_path}" if step is None else f"{self.checkpoint_path}/step_{int(step):05}"
        os.makedirs(actual_checkpoint_path, exist_ok=True)


        # Save
        H_df = pd.DataFrame(H.T, index=self.sample_list, columns=latent_columns)
        H_df.to_parquet(f"{actual_checkpoint_path}/H.parquet")
        for d, Wd in enumerate(Ws): pd.DataFrame(Wd, index=self.omics_features[d], columns=latent_columns).to_parquet(f"{actual_checkpoint_path}/W{d}.parquet")


        # Log
        if step is not None: logging.info(f"Saved model to: {actual_checkpoint_path}")
        else: logging.info(f"Saved checkpoint for step #{step} to: {actual_checkpoint_path}")
