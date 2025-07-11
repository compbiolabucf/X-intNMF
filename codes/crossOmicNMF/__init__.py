# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: __init__.py
# Date: 2024/09/12 15:54:51
# Description: Main script for running the CrossOmicDataInt package
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

from typing import List, Tuple, Union, Literal, Any, Callable, Dict
from sklearn.linear_model import Lasso
from filelock import FileLock
import numpy as np
import pandas as pd
import logging
import mlflow



class XIntNMF:    
    """
        Class to solve the cross-omics, multi-omics layers integration problem 

        Data:
        ---
        - `omics_layers`: List[np.ndarray]
            A list of omics layers matrices of shape (m_d, N). Now denotes as X_d
        - `cross_omics_interaction`: Dict[Tuple[int, int], np.ndarray]
            A dictionary of cross-omics interaction matrices of shape (m_p, m_q) with key as (p, q) where p, q are the interaction matrix of p-th and q-th omics layers
        - `internal_interaction`: List[np.ndarray]
            A list of on-diagonal interaction matrices of shape (m_d, m_d)
        

        Hyperparameters:
        ---
        - `num_clusters`: int
            The number of latent variables or clusters to initialize the W matrices. Now denotes as k
        - `cross_omics_alpha`: float
            The hyperparameter to control the graph regularization term. Now denotes as alpha
        - `sparsity_W_control_betas`: Union[float, List[float]]
            The hyperparameter to control the sparsity of W matrices. If a single value is passed, the same value will be used for all W matrices. If a list is passed, the value will be used for each W matrix. Now denotes as betas
        - `lasso_control_gammas`: Union[np.array, List, float]
            The L1 regularization parameter for each sample. If a single value is passed, the same value will be used for all samples. If a list is passed, the value will be used for each sample. Now denotes as gammas
        
            
        Control parameters:
        ---
        - `similarity_method`: Union[Callable, None]
            The similarity metric to calculate the similarity matrix. If None, the default metric is Pearson correlation coefficient. The method should take the omics layers matrix as input and return the similarity matrix of shape (feature, feature), or (m, m)
        - `max_iter`: int
            The maximum number of iterations to run the algorithm
        - `tol`: float
            The tolerance to stop the algorithm
        - `verbose`: bool
            Whether to print the debug information or not
        - `gpu`: Union[int, None]
            The GPU device to use. Default is None, which means using CPU
    """



    # Data - Directly from the input
    Xd_source: List[np.ndarray]                                 # Omics layers
    cross_omics_interaction: Dict[Tuple[int, int], np.ndarray]  # Off-diagonal/Cross-omics interaction. Will contains internal interaction as well

    # Hyperparameters - Directly from the input
    k: int                                                      # Number of latent variables or clusters    
    alpha: float                                                # Graph regularization term control
    betas: Union[np.array, List, float]                         # Sparsity control for W matrices
    gammas: Union[np.array, List, float]                        # L1 regularization & sparsity parameter for each sample

    # Control parameters - Directly from the input
    max_iter: int                                               # Maximum number of iterations
    tol: float                                                  # Tolerance to stop the algorithm
    verbose: bool                                               # Whether to print the debug information or not
    similarity_method: Union[Callable, None]                    # The similarity metric to calculate the similarity matrix. If None, the default metric is Pearson correlation coefficient. The method should take the omics layers matrix as input and return the similarity matrix of shape (feature, feature), or (m, m)
    device: Union[str, None] = None                             # Device to use. Default is None
    backend: Literal['numpy', 'cupy', 'pytorch'] = 'numpy'      # Backend for matrix computation. Default is numpy
    mlflow_enabled: bool                                        # Whether to enable MLFlow logging or not

    # Internal variables, inferred from input
    D: int                                                      # Number of omics layers
    N: int                                                      # Number of samples
    M: int                                                      # Number of features in all omics layers
    m: List[int]                                                # Number of features in each omics layer
    E: List[List[np.ndarray]]                                   # Cross-omics interaction matrix block, as list of list form for better normalization and concatenation
    

    def __init__(
        self,
        
        # Data - Directly from the input
        omics_layers: List[np.ndarray],                                 # Omics layers
        cross_omics_interaction: Dict[Tuple[int, int], np.ndarray],     # Off-diagonal/Cross-omics interaction

        # Hyperparameters - Directly from the input
        k: int,                                                         # Number of latent variables or clusters    
        alpha: float,                                                   # Graph regularization term control
        betas: Union[np.array, List, float],                            # Sparsity control for W matrices. The sample will be automatically broadcasted to all if a single value is passed
        gammas: Union[np.array, List, float],                           # L1 regularization & sparsity parameter for each sample. The sample will be automatically broadcasted to all if a single value is passed

        # Control parameters - Directly from the input
        max_iter: int,                                                  # Maximum number of iterations
        tol: float,                                                     # Tolerance to stop the algorithm
        verbose: bool = False,                                          # Whether to print the debug information or not 
        gpu: Union[int, None] = None,                                   # GPU device to use. Default is None
        backend: Literal['numpy', 'cupy', 'pytorch'] = 'numpy',         # Backend for matrix computation. Default is numpy
        mlflow_enable: bool = True,                                     # Whether to enable MLFlow logging or not
    ):
        """
            Initialize the XIntNMF class with the following parameters:
            - `omics_layers` (`List[np.ndarray]`): 
                Omics layers.
            - `cross_omics_interaction` (`Dict[Tuple[int, int], np.ndarray]`): 
                Off-diagonal/Cross-omics interaction.
            - `k` (`int`):  
                Number of latent variables or clusters.
            - `alpha` (`float`): 
                Graph regularization term control.
            - `betas` (`Union[np.array, List, float]`): 
                Sparsity control for W matrices. The sample will be automatically broadcasted to all if a single value is passed.
            - `gammas` (`Union[np.array, List, float]`): 
                L1 regularization & sparsity parameter for each sample. The sample will be automatically broadcasted to all if a single value is passed.
            - `max_iter` (`int`): 
                Maximum number of iterations.
            - `tol` (`float`): 
                Tolerance to stop the algorithm.
            - `verbose` (`bool`, optional): 
                Whether to print the debug information or not. Default is False.
            - `gpu` (`Union[int, None]`, optional): 
                GPU device to use. Default is None.
            - `backend` (`Literal['numpy', 'cupy', 'pytorch']`, optional): 
                Backend for matrix computation. Default is numpy.
            - `mlflow_enable` (`bool`, optional): 
                Whether to enable MLFlow logging or not. Default is True.
        """


        # Direct input
        self.Xd_source = omics_layers
        self.cross_omics_interaction = cross_omics_interaction

        self.k = k
        self.alpha = alpha

        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.backend = backend
        self.device = (f'cuda:{gpu}' if isinstance(gpu, int) else gpu) if gpu is not None else 'cpu'
        self.mlflow_enabled = mlflow_enable


        # Inferred from input
        self.D = len(omics_layers)
        self.N = omics_layers[0].shape[1]
        self.m = [omic.shape[0] for omic in omics_layers]
        self.M = sum(self.m)
    
        # Auto-broadcasting hyperparameters beta and gamma
        self.betas = [np.float64(betas)] * self.D if isinstance(betas, (int, float)) else betas
        self.gammas = [np.float64(gammas)] * self.N if isinstance(gammas, (int, float)) else gammas

        # Raise unmatched length error if the length of beta and gamma is not matched
        if len(self.betas) != self.D: raise ValueError(f"Length of beta is not matched with the number of omics layers. Expected {self.D} but got {len(self.betas)}")
        if len(self.gammas) != self.N: raise ValueError(f"Length of gamma is not matched with the number of samples. Expected {self.N} but got {len(self.gammas)}")

        # Debug: Print the log
        logging.info(f"Initialized CrossOmicDataInt with {self.D} omics layers, {self.N} samples, {self.M} features, {self.k} clusters, alpha={alpha}, betas={betas}, gammas={gammas}, max_iter={self.max_iter}, tol={self.tol}")

        # Backend selection
        if self.backend == 'numpy':
            from ._math import IterativeSolveWdsAndH
            self.solver = IterativeSolveWdsAndH
        elif self.backend == 'cupy':
            from ._math_cupy import IterativeSolveWdsAndH_CuPy
            self.solver = IterativeSolveWdsAndH_CuPy
        elif self.backend == 'pytorch':
            from ._math_pytorch import IterativeSolveWdsAndH_PyTorch
            self.solver = IterativeSolveWdsAndH_PyTorch
            

        # MLFlow logging
        if self.mlflow_enabled:
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("betas", betas)
            mlflow.log_param("gammas", gammas)
            mlflow.log_param("max_iter", self.max_iter)
            mlflow.log_param("tol", self.tol)
            mlflow.log_param("verbose", self.verbose)

            mlflow.log_param("Number of omics layers", self.D)
            mlflow.log_param("Sample size", self.N)
            mlflow.log_param("Omics layers feature size", self.m)
            mlflow.log_param("Latent size", self.k)




    def debug(self, message: str):
        if self.verbose: logging.info(message)

    
    


    from ._supports import compute_similarity
    from ._supports import divisor_func_selection
    from ._supports import norm_func_selection
    from ._supports import threshold_cutoff
    from ._supports import density_func_selection
    from ._supports import density_agg_func_selection
    from ._supports import convert_block_matrix_to_table_of_matrices

    
    from ._operations import ComposeRawBaseline
    from ._operations import PrenormalizeData
    from ._operations import InitializeAndConcatenateInteraction
    from ._operations import InitializeBalancedCutoff
    from ._operations import InitializeWd
    from ._operations import LassoInitializeH


    from ._debug import PrenormalizeDebug
    from ._debug import nullityCheck
    from ._debug import negativeCheck
    


    def solve(
        self,

        # Pre-normalization
        pre_normalize: bool = True,
        pre_normalize_method: Union[Callable, Literal["l2", "l1", "max", "min", "mean", "median", "sum", "passthru"]] = "max",
        pre_normalize_orientation: Literal["row", "column", "all"] = "row",

        # Initialize and concatenate interaction matrices
        internal_interaction_init_methods: Union[Callable, None] = None,
        internal_interaction_get_abs_value: bool = True,

        # Balancing cutoff
        balancing_cutoff_density_measure: Union[Callable, Literal['density', 'gini', 'rank']] = "density",
        balancing_cutoff_compare_with: Literal['row', 'column', 'rowcol', 'off_diag'] = "off_diag",
        balancing_cutoff_density_agg_method: Literal['mean', 'median', 'max', 'min'] = "mean",

        # Initialize Wd
        initialize_Wd_method: Union[Callable, Literal["random", "nmf"]] = "nmf",

        # Lasso: Use CV?
        use_cross_validation: bool = True,

        # Evaluation:
        run_mode: Literal['full', 'nmf_lasso_only', 'norm_baseline', 'raw_baseline'] = 'full',
        
        # Test AUC
        additional_tasks: Union[Callable, None] = None,
        additional_tasks_interval: int = 50,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
            Solve the cross-omics, multi-omics layers integration problem

            Inputs:
            ---
            - `pre_normalize` (`bool`): 
                Whether to pre-normalize the data. Default is True.
            - `pre_normalize_method` (`Union[Callable, Literal]`): 
                Method for pre-normalization. Default is "max".
            - `pre_normalize_orientation` (`Literal`): 
                Orientation for pre-normalization. Default is "row".
            - `internal_interaction_init_methods` (`Union[Callable, None]`): 
                Methods to initialize interaction matrices. Default is None.
            - `internal_interaction_get_abs_value` (`bool`): 
                Whether to get absolute value for interaction matrices. Default is True.
            - `balancing_cutoff_density_measure` (`Union[Callable, Literal]`): 
                Density measure for balancing cutoff. Default is "density".
            - `balancing_cutoff_compare_with` (`Literal`): 
                Comparison method for balancing cutoff. Default is "off_diag".
            - `balancing_cutoff_density_agg_method` (`Literal`): 
                Aggregation method for density in balancing cutoff. Default is "mean".
            - `initialize_Wd_method` (`Union[Callable, Literal]`): 
                Method to initialize Wd matrices. Default is "nmf".
            - `use_cross_validation` (`bool`): 
                Whether to use cross-validation for Lasso. Default is True.
            - `run_mode` (`Literal`): 
                Mode to run the solver. Default is 'full'.
            - `additional_tasks` (`Union[Callable, None]`): 
                Additional tasks to perform during solving. Default is None.
            - `additional_tasks_interval` (`int`): 
                Interval for additional tasks. Default is 50.

            Returns:
            `Tuple[List[np.ndarray], np.ndarray]`: The solved W matrices and H matrix. W matrices are of shape (m_d, k) and H matrix is of shape (k, N), denotes omics factors and sample factor respectively.
        """
        self.Xd = self.Xd_source
        # -----------------------------------------------------------------------------------------------
        # Compose raw baseline if needed
        # -----------------------------------------------------------------------------------------------
        if run_mode == 'raw_baseline':
            Wd = []

            for x in self.Xd:
                logging.fatal(f"Xd shape: {x.shape}")

            H = self.ComposeRawBaseline()
            return Wd, H


        # -----------------------------------------------------------------------------------------------
        # 0.Pre-normalize the data
        # -----------------------------------------------------------------------------------------------
        if pre_normalize:
            logging.warning("[0/6] Pre-normalizing the data...")
            self.PrenormalizeData(
                method = pre_normalize_method,
                orientation = pre_normalize_orientation
            )
            logging.warning("Pre-normalization completed")


            if run_mode == 'norm_baseline':
                Wd = []
                H = self.ComposeRawBaseline()
                return Wd, H






        # -----------------------------------------------------------------------------------------------
        # 1+2.Initialize and concatenate the interaction matrices
        # -----------------------------------------------------------------------------------------------
        logging.warning("[1+2/6] Initializing and concatenating the interaction matrices...")
        self.InitializeAndConcatenateInteraction(
            internal_interaction_init_methods,
            internal_interaction_get_abs_value
        )
        logging.warning("Initialization and concatenation completed")






        # -----------------------------------------------------------------------------------------------
        # 3.Cut-off and balance the internal interaction matrices w/ density measure
        # -----------------------------------------------------------------------------------------------
        logging.warning("[3/6] Balancing the cutoff...")
        self.InitializeBalancedCutoff(
            density_measure = balancing_cutoff_density_measure,
            compare_with = balancing_cutoff_compare_with,
            density_agg_method = balancing_cutoff_density_agg_method
        )
        logging.warning("Balancing cutoff completed")




        logging.warning("[4/6] Initializing W matrices...")
        # -----------------------------------------------------------------------------------------------
        # 4.Initialize W matrices - Required lock. If two processes are running at the same time will risk freezing
        # -----------------------------------------------------------------------------------------------
        Wds = self.InitializeWd(
            method = initialize_Wd_method,
            silence_verbose_override = True
        )
        for i, Wd in enumerate(Wds):
            logging.warning(f"Initialized Wd[{i}] with shape {Wd.shape}")
            # logging.warning(Wd)
        logging.warning("Initialization of W matrices completed")






        # -----------------------------------------------------------------------------------------------
        # 5.Solve the H matrix using Lasso - Required lock
        # -----------------------------------------------------------------------------------------------
        logging.warning("[5/6] Initializing H using Lasso...")
        H = self.LassoInitializeH(
            initialized_Wds = Wds,
            use_cross_validation = use_cross_validation,
        )
        logging.warning(f"Initialized H completed with shape {H.shape}")

        # logging.warning("Lock released")





        # -----------------------------------------------------------------------------------------------
        # 6.Iteratively solve the W matrices
        # -----------------------------------------------------------------------------------------------
        if run_mode == 'nmf_lasso_only':
            return Wds, H

        
        logging.warning("[6/6] Iteratively solving H and Ws...")
        
        res_Wds, res_H = self.solver(
            self,
            initialized_Wds = Wds,
            initialized_H = H,
            additional_tasks = additional_tasks,
            additional_tasks_interval = additional_tasks_interval,
        )

        return res_Wds, res_H
        

        

    


# Content and code by bu1th4nh. Written with dedication in the University of Central Florida and the Magic Kingdom.
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
                                                                                                                                       
