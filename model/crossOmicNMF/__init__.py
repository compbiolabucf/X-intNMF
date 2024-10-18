# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành (@bu1th4nh)
# Title: main.py
# Date: 2024/09/12 15:54:51
# Description: Main script for running the CrossOmicDataInt package
# 
# (c) bu1th4nh. All rights reserved
# -----------------------------------------------------------------------------------------------

from typing import List, Tuple, Union, Literal, Any, Callable, Dict
from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd
import logging
import mlflow



class SimilarSampleCrossOmicNMF:
    """
        Class to solve the cross-omics, multi-omics layers integration problem 

        Data:
        - `omics_layers`: List[np.ndarray]
            A list of omics layers matrices of shape (m_d, N). Now denotes as X_d
        - `cross_omics_interaction`: Dict[Tuple[int, int], np.ndarray]
            A dictionary of cross-omics interaction matrices of shape (m_p, m_q) with key as (p, q) where p, q are the interaction matrix of p-th and q-th omics layers
        - `internal_interaction`: List[np.ndarray]
            A list of on-diagonal interaction matrices of shape (m_d, m_d)
        

        Hyperparameters:
        - `num_clusters`: int
            The number of latent variables or clusters to initialize the W matrices. Now denotes as k
        - `cross_omics_alpha`: float
            The hyperparameter to control the graph regularization term. Now denotes as alpha
        - `sparsity_W_control_betas`: Union[float, List[float]]
            The hyperparameter to control the sparsity of W matrices. If a single value is passed, the same value will be used for all W matrices. If a list is passed, the value will be used for each W matrix. Now denotes as betas
        - `lasso_control_gammas`: Union[np.array, List, float]
            The L1 regularization parameter for each sample. If a single value is passed, the same value will be used for all samples. If a list is passed, the value will be used for each sample. Now denotes as gammas
        
            
        Control parameters:
        - `similarity_method`: Union[Callable, None]
            The similarity metric to calculate the similarity matrix. If None, the default metric is Pearson correlation coefficient. The method should take the omics layers matrix as input and return the similarity matrix of shape (feature, feature), or (m, m)
        - `max_iter`: int
            The maximum number of iterations to run the algorithm
        - `tol`: float
            The tolerance to stop the algorithm
        - `verbose`: bool
            Whether to print the debug information or not
    """



    # Data - Directly from the input
    Xd_source: List[np.ndarray]                                        # Omics layers
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
    ):
        # Direct input
        self.Xd_source = omics_layers
        self.cross_omics_interaction = cross_omics_interaction

        self.k = k
        self.alpha = alpha

        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose


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

        # MLFlow logging
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

    
    from ._math import objective_function
    from ._math import update


    from ._supports import compute_similarity
    from ._supports import divisor_func_selection
    from ._supports import norm_func_selection
    from ._supports import threshold_cutoff
    from ._supports import density_func_selection
    from ._supports import density_agg_func_selection
    from ._supports import convert_block_matrix_to_table_of_matrices

    
    from ._operations import PrenormalizeData
    from ._operations import InitializeAndConcatenateInteraction
    from ._operations import InitializeBalancedCutoff
    from ._operations import InitializeWd
    from ._operations import LassoSolveH
    from ._operations import IterativeSolveWdsAndH


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
        no_iterations: bool = False, # Only for evaluation, output the initial W matrices and H matrix


    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
            Solve the cross-omics, multi-omics layers integration problem
        """
        # -----------------------------------------------------------------------------------------------
        # 0.Pre-normalize the data
        # -----------------------------------------------------------------------------------------------
        self.Xd = self.Xd_source
        if pre_normalize:
            logging.warning("[0/6] Pre-normalizing the data...")
            self.PrenormalizeData(
                method = pre_normalize_method,
                orientation = pre_normalize_orientation
            )
            logging.warning("Pre-normalization completed")









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






        # -----------------------------------------------------------------------------------------------
        # 4.Initialize W matrices
        # -----------------------------------------------------------------------------------------------
        logging.warning("[4/6] Initializing W matrices...")
        Wds = self.InitializeWd(
            method = initialize_Wd_method,
            silence_verbose_override = True
        )
        for i, Wd in enumerate(Wds):
            logging.warning(f"Initialized Wd[{i}] with shape {Wd.shape}")
            logging.warning(Wd)
        logging.warning("Initialization of W matrices completed")






        # -----------------------------------------------------------------------------------------------
        # 5.Solve the H matrix using Lasso
        # -----------------------------------------------------------------------------------------------
        logging.warning("[5/6] Solving the H matrix using Lasso...")
        H = self.LassoSolveH(
            initialized_Wds = Wds,
            use_cross_validation = use_cross_validation,
        )
        logging.warning(f"Solved H with shape {H.shape}, H:\n{H}")
        logging.warning("Solving H matrix completed")





        # -----------------------------------------------------------------------------------------------
        # 6.Iteratively solve the W matrices
        # -----------------------------------------------------------------------------------------------
        if no_iterations: return Wds, H

        
        logging.warning("[6/6] Iteratively solving the W matrices...")
        res_Wds, res_H = self.IterativeSolveWdsAndH(
            initialized_Wds = Wds,
            initialized_H = H
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
                                                                                                                                       
