# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành (@bu1th4nh)
# Title: _operations.py
# Date: 2024/09/17 14:47:01
# Description: 
# 
# (c) bu1th4nh. All rights reserved
# -----------------------------------------------------------------------------------------------


import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import NMF
from typing import List, Tuple, Union, Literal, Any, Callable, Dict
from sklearn.linear_model import Lasso, LassoCV, MultiTaskLasso, MultiTaskLassoCV

def PrenormalizeData(
    self,
    method: Union[Callable, Literal["l2", "l1", "max", "min", "mean", "median", "sum", "passthru"]] = "max",
    orientation: Literal["row", "column", "all"] = "row",
):
    """
        Pre-normalize the omic layers matrix

        Parameters
        ----------
        method: Union[Callable, Literal["l2", "l1", "max", "min", "mean", "median", "sum", "passthru"]]
            The normalization method to normalize the omic layers matrix. If `passthru` is chosen, the matrix will not be normalized
        orientation: Literal["row", "column", "all"]
            The orientation to normalize the omic layers matrix
    """
    norm_func = self.norm_func_selection(method, orientation)
    for d, X in enumerate(self.Xd): self.Xd[d] = norm_func(X)

    logging.info("Nullity and nonnegativity check after pre-normalization...")
    for d, X in enumerate(self.Xd): self.nullityCheck(X, f"Omic layer #{d}")
    for d, X in enumerate(self.Xd): self.negativeCheck(X, f"Omic layer #{d}")
    logging.info("Pre-normalization completed")






def InitializeAndConcatenateInteraction(
    self,
    methods: Union[Callable, None] = None,
    get_abs_value: bool = True,
):
    """
        Initialize similarity matrix for each omic layer and concatenate them with cross-omics interaction matrix

        Parameters
        ----------
        methods: Callable, optional
            The method to calculate similarity matrix for each omic layer
    """

    for (d, X) in enumerate(self.Xd): self.cross_omics_interaction[(d, d)] = self.compute_similarity(X, methods, get_abs_value)


    # Construct the concatenated similarity matrix
    self.E = []
    for i in range(len(self.m)):
        row = []
        for j in range(len(self.m)):
                if (i, j) in self.cross_omics_interaction:
                    row.append(self.cross_omics_interaction[(i, j)])
                elif (j, i) in self.cross_omics_interaction:
                    row.append(self.cross_omics_interaction[(j, i)].T)
                else:
                    row.append(np.zeros((self.m[i], self.m[j])))
        self.E.append(row)

    

    # Debug: Output the concatenated similarity matrix shape
    logging.info(f"Concatenated similarity matrix shape: {np.block(self.E).shape}")
    for h_block in self.E:
        for v_block in h_block:
            logging.info(f"Block shape: {v_block.shape}")

                

    # Double check the concatenated similarity matrix shape
    logging.info("Shape checking the concatenated similarity matrix...")
    for d1 in range(self.D):
        for d2 in range(self.D):
            # Check if each blocks on the same row have the same number of rows. We do that by comparing the number of rows of the first block with the rest
            assert self.E[d1][d2].shape[0] == self.E[d1][0].shape[0], f"Invalid row shape of concatenated similarity matrix at row {d1}, column {d2}) detected ({self.E[d1][d2].shape[0]} != {self.E[d1][0].shape[0]})"

            # Check if each blocks on the same column have the same number of columns. Since E has equal number of rows and columns, we only need to swap d1 and d2
            assert self.E[d2][d1].shape[1] == self.E[0][d1].shape[1], f"Invalid column shape of concatenated similarity matrix at row {d2}, column {d1}) detected ({self.E[d2][d1].shape[1]} != {self.E[0][d1].shape[1]})"

    
    # Nullity check
    logging.info("Nullity check...")
    for d1, row in enumerate(self.E):
        for d2, element in enumerate(row):
            self.nullityCheck(element, f"Similarity matrix ({d1}, {d2})")
    
    # Non-negativity check
    logging.info("Non-negativity check...")
    for d1, row in enumerate(self.E):
        for d2, element in enumerate(row):
            self.negativeCheck(element, f"Similarity matrix ({d1}, {d2})")
    
    


 
    
def InitializeBalancedCutoff(
    self,
    density_measure: Union[Callable, Literal['density', 'gini', 'rank']],
    compare_with: Literal['row', 'column', 'rowcol', 'off_diag'],
    density_agg_method: Literal['mean', 'median', 'max', 'min'],
):
    """
        Transform internal similarity matrix to a 0-1 matrix with same sparsity measure as the chosen group of cross-omics interaction matrices. The chosen group of cross-omics interaction matrices can be one or some of the following:
            - 'row': Compare the sparsity of the similarity matrix with the row-wise cross-omics interaction matrices,
            - 'column': Compare the sparsity of the similarity matrix with the column-wise cross-omics interaction matrices.
            - 'rowcol': Compare the sparsity of the similarity matrix with all row-wise and column-wise cross-omics interaction matrices.
            - 'off_diag': Compare the sparsity of the similarity matrix with all off-diagonal cross-omics interaction matrices. 
        
        After choosing the group of cross-omics interaction matrices, the algorithm will calculate the sparsity measure of the chosen group of cross-omics interaction matrices using the given sparsity measure, then aggregate the sparsity measure of the chosen group of cross-omics interaction matrices using the following aggregation methods:
            - 'mean': Calculate the mean of the sparsity measure of the chosen group of cross-omics interaction matrices
            - 'median': Calculate the median of the sparsity measure of the chosen group of cross-omics interaction matrices
            - 'max': Calculate the maximum of the sparsity measure of the chosen group of cross-omics interaction matrices
            - 'min': Calculate the minimum of the sparsity measure of the chosen group of cross-omics interaction matrices

        The sparsity measure can be one of the following:
            - 'density': The number of non-zero elements in the matrix
            - 'gini': The Gini coefficient of the matrix
            - 'rank': The rank of the matrix

        After aggregating the sparsity measure of the chosen group of cross-omics interaction matrices, the algorithm will use binary search to find the cutoff value that makes the sparsity measure of the transformed matrix equal to the aggregated sparsity measure of the chosen group of cross-omics interaction matrices using the given sparsity measure
        

        Parameters
        ----------
        sparsity_measure: Callable or Literal['density', 'gini', 'rank']
            The sparsity measure to compare the sparsity of the similarity matrix with the cross-omics interaction matrix

        compare_with: Literal['row', 'column', 'rowcol', 'off_diag']
            The group of cross-omics interaction matrices to compare the sparsity of the similarity matrix with

        density_agg_method: Literal['mean', 'median', 'max', 'min']
            The aggregation method to aggregate the sparsity measure of the chosen group of cross-omics interaction matrices
    """

    # Select the sparsity measure function
    density_method = self.density_func_selection(density_measure)

    # Compute the sparsity measure of the chosen group of cross-omics interaction matrices. The diagonal elements are ignored and set to NaN as an indicator for the algorithm to ignore
    density = np.zeros((self.D, self.D))
    for d1 in range(self.D):
        for d2 in range(self.D):
            if d1 == d2: density[d1, d2] = np.nan
            else: density[d1, d2] = density_method(self.E[d1][d2])

    # Aggregate the sparsity measure of the chosen group of cross-omics interaction matrices. The functions already ignored NaN values
    desired_density = []
    density_agg_method = self.density_agg_func_selection(density_agg_method)
    for d in range(self.D): 
        if compare_with == 'row': desired_density.append(density_agg_method(density[d, :]))
        elif compare_with == 'column': desired_density.append(density_agg_method(density[:, d]))
        elif compare_with == 'rowcol': desired_density.append(density_agg_method(np.concatenate([density[d, :], density[:, d]])))
        elif compare_with == 'off_diag': desired_density.append(density_agg_method(density[np.triu_indices(self.D, k=1)]))
        else: raise ValueError(f"Invalid value for compare_with parameter: {compare_with}")
        

    # Binary search for the cutoff value that makes the sparsity measure of the transformed matrix equal to the aggregated sparsity measure of the chosen group of cross-omics interaction matrices
    for d in range(self.D):
        max_cutoff = np.max(self.E[d][d])
        min_cutoff = np.min(self.E[d][d])
        logging.info(f"Balanced cutoff for omic layer {d}, max cutoff: {max_cutoff}, min cutoff: {min_cutoff}")

        cutoff = None
        iteration = 0
        while np.abs(max_cutoff - min_cutoff) >= 1e-8:
            iteration += 1
            cutoff = (max_cutoff + min_cutoff) / 2
            transformed_matrix = self.threshold_cutoff(self.E[d][d], cutoff)
            transformed_density = density_method(transformed_matrix)

            logging.info(f"Iter #{iteration:03}, Min = {min_cutoff:.4f}, Max = {max_cutoff:.4f}, Cutoff = {cutoff:.4f}, Density = {transformed_density:.10f}, Desired = {desired_density[d]:.10f}, Diff = {np.abs(transformed_density - desired_density[d]):.10f}")


            if transformed_density < desired_density[d]: max_cutoff = cutoff
            else: min_cutoff = cutoff
        
        self.E[d][d] = self.threshold_cutoff(self.E[d][d], cutoff)
        logging.info(f"Balanced cutoff for omic layer {d} is {cutoff}")

    
        



def InitializeWd(
    self,
    method = Union[Callable, Literal["random", "nmf"]],
    silence_verbose_override: bool = False,
) -> List[np.ndarray]:
    """
        Initialize the W matrices for each omic data (4)
        
        Input
        -----
        `method`: Callable or Literal["random", "nmf"]
            The method to initialize the W matrices. The callable method should take an omic data as input and return the W matrix of shape (m_d, k)
            
        `silence_verbose_override`: bool
            Silence the verbose output of the method. This will override the verbose parameter of the class


        Output
        ------
        W: List[np.ndarray]
            A list of W matrices of shape (m_d, k)
    """


    # Compute
    Wds = []
    if callable(method):
        for omic in self.Xd:
            Wds.append(method(omic))
        # return Wds
    elif method == 'nmf':
        Wds = []
        for omic in self.Xd:
            nmf_model = NMF(
                n_components = self.k, 
                init         = 'nndsvd', 
                alpha_W      = 1/np.sqrt(omic.shape[0]),
                tol          = self.tol,
                max_iter     = self.max_iter,
                verbose      = int(self.verbose and (not silence_verbose_override)),
            )
            # Docs: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
            # NMF: X(feature, sample) -> W~fit_transform(feature, latent) + H~components(latent, sample)
            Wds.append(nmf_model.fit_transform(omic))
        # return Wds
    else: Wds = [np.random.rand(m, self.k) for m in self.m]

    # Nullity check
    logging.info("Nullity check...")
    for d, W in enumerate(Wds): self.nullityCheck(W, f"W matrix #{d}")

    # Non-negativity check
    logging.info("Non-negativity check...")
    for d, W in enumerate(Wds): self.negativeCheck(W, f"W matrix #{d}")

    return Wds



    


def LassoSolveH(
    self,
    initialized_Wds: List[np.ndarray], 
    use_cross_validation: bool = False,
) -> np.ndarray:
    """
        Solve the H matrix with fixed W matrices (5)
        
        Input
        -----
        `omics_layers`: List[np.ndarray]
            A list of omics layers matrices of shape (m_d, N)
        `initialized_Wds`: List[np.ndarray]
            A list of initialized W matrices of shape (m_d, k). The cluster size will be self-inferred from the shape of the W matrices
        `use_cross_validation`: bool = False
            Whether to use cross-validation to find the optimal gamma value for the Lasso regression. If True, gamma will be inferred from cross-validation. If False, gamma will be inferred from the class attribute `gammas`
        


        Output
        ------
        H: np.ndarray
            The H matrix of shape (k, N)
    """


    # Concat all omics layers matrix along the column axis, aka big_X = [X_1; X_2; ...; X_D] with shape (m_1 + m_2 + ... + m_D, N)
    big_X = np.concatenate(self.Xd, axis=0)
    logging.info(f"Big X shape: {big_X.shape}")
    
    # Concat all Ws data matrix along the column axis, aka big_W = [W_1; W_2; ...; W_D] with shape (m_1 + m_2 + ... + m_D, k)
    big_W = np.concatenate(initialized_Wds, axis=0)
    logging.info(f"Big W shape: {big_W.shape}")
    logging.fatal(f"Big W: \n{big_W}")


    # Initialize H
    H_coeffs = []

    # Solve individual column of H 
    for index, gamma in tqdm(enumerate(self.gammas), desc="Solving H matrix"):

        if use_cross_validation:
            lasso = LassoCV(
                fit_intercept = False, 
                positive = True, 
                cv = 5, 
                verbose = False,
                n_jobs = -1,
            );
        else:
            lasso = Lasso(
                alpha = gamma, 
                fit_intercept = False, 
                positive = True, 
                warm_start = True
            );
        lasso.fit(
            big_W,
            big_X[:, index],
        )
        H_coeffs.append(np.array(lasso.coef_.T))

        # logging.info(f"Column {index}/{big_X.shape[1]}")
        # logging.info(f"  big_X[:, index]: \n{big_X[:, index]}")
        # logging.info(f"  Coefficients: {lasso.coef_}")
        # logging.info(f"  Intercept: {lasso.intercept_}")

    # Construct H
    H = np.array(H_coeffs).T
    logging.info(f"H shape: {H.shape}")
    logging.info(f"H density: {self.density_func_selection('density')(H)}")

    # Nullity check
    logging.info("Nullity check...")
    self.nullityCheck(H, "solved H matrix")

    # Non-negativity check
    logging.info("Non-negativity check...")
    self.negativeCheck(H, "solved H matrix")

    return H






def NultitaskLassoSolveH(
    self,
    initialized_Wds: List[np.ndarray], 
    use_cross_validation: bool = False,
) -> np.ndarray:
    """
        Solve the H matrix with fixed W matrices (5)
        
        Input
        -----
        `omics_layers`: List[np.ndarray]
            A list of omics layers matrices of shape (m_d, N)
        `initialized_Wds`: List[np.ndarray]
            A list of initialized W matrices of shape (m_d, k). The cluster size will be self-inferred from the shape of the W matrices
        `use_cross_validation`: bool = False
            Whether to use cross-validation to find the optimal gamma value for the Lasso regression. If True, gamma will be inferred from cross-validation. If False, gamma will be inferred from the class attribute `gammas`
        


        Output
        ------
        H: np.ndarray
            The H matrix of shape (k, N)
    """


    # Concat all omics layers matrix along the column axis, aka big_X = [X_1; X_2; ...; X_D] with shape (m_1 + m_2 + ... + m_D, N)
    big_X = np.concatenate(self.Xd, axis=0)
    logging.info(f"Big X shape: {big_X.shape}")
    
    # Concat all Ws data matrix along the column axis, aka big_W = [W_1; W_2; ...; W_D] with shape (m_1 + m_2 + ... + m_D, k)
    big_W = np.concatenate(initialized_Wds, axis=0)
    logging.info(f"Big W shape: {big_W.shape}")
    logging.fatal(f"Big W: \n{big_W}")


    # Initialize H
    H_coeffs = []

    # Solve individual column of H 
    for index, gamma in enumerate(self.gammas):
        lasso = Lasso(
            alpha = gamma, 
            fit_intercept = False, 
            positive = True, 
            warm_start = True
        );
        lasso.fit(
            big_W,
            big_X[:, index],
        )
        H_coeffs.append(np.array(lasso.coef_.T))

        logging.info(f"Column {index}/{big_X.shape[1]}")
        logging.info(f"  big_X[:, index]: \n{big_X[:, index]}")
        logging.info(f"  Coefficients: {lasso.coef_}")
        logging.info(f"  Intercept: {lasso.intercept_}")

    # Construct H
    H = np.array(H_coeffs).T
    logging.info(f"H shape: {H.shape}")

    # Nullity check
    logging.info("Nullity check...")
    self.nullityCheck(H, "solved H matrix")

    # Non-negativity check
    logging.info("Non-negativity check...")
    self.negativeCheck(H, "solved H matrix")

    return H






def IterativeSolveWds(
    self,
    initialized_Wds:        List[np.ndarray], 
    H:                      np.ndarray, 
) -> List[np.ndarray]:
    """
        Iteratively solve the W matrices with fixed H matrix (6)
        
        Input
        -----
        `H`: np.ndarray
            The H matrix of shape (k, N)
        `cross_omic_similarity`: List[List[np.ndarray]]
            The concatenated similarity matrix block of shape (M, M)

        Output
        ------
        W: List[np.ndarray]
            A list of W matrices of shape (m_d, k)
    """

    # Construct the omic indices for matrix splitting
    omics_indices = np.cumsum(self.m)[:-1] # Drop the final index
    
    # Construct the normalized similarity matrix
    E = np.block(self.E)
    D = np.diag(1 / np.sqrt(np.sum(E, axis=1)))
    E_normalized = D @ E @ D
    E = self.convert_block_matrix_to_table_of_matrices(E_normalized, omics_indices)

    # Construct the degree matrix
    degree_block = self.convert_block_matrix_to_table_of_matrices(np.eye(self.M), omics_indices)

    # Debug: Output the split indices and the shape of the degree block
    logging.info(f"Split indices: {omics_indices}")
    logging.info(f"Degree block - Total shape: {np.block(degree_block).shape}")
    logging.info(f"Degree block - individual shape:")
    for hblk in degree_block:
        logging.info("  ".join(f"{blk.shape}" for blk in hblk))
    
    # Debug: Output the shape of the similarity block
    logging.info(f"Similarity block - total shape: {np.block(E).shape}")
    logging.info(f"Degree block - individual shape:")
    for hblk in self.E:
        logging.info("  ".join(f"{blk.shape}" for blk in hblk))
    
    # Iteratively solve the W matrices
    Ws = initialized_Wds
    iteration = 0
    curr_obj = self.objective_function(self.Xd, Ws, H, E, degree_block, self.alpha, self.betas, self.gammas)


    while True:
        Ws = self.update_Ws(self.Xd, Ws, H, E, degree_block, self.alpha, self.betas)
        next_obj = self.objective_function(self.Xd, Ws, H, E, degree_block, self.alpha, self.betas, self.gammas)

        iteration += 1
        logging.info(f"Iteration {iteration}: Objective function = {next_obj}")

        if np.abs(next_obj - curr_obj) < self.tol or iteration >= self.max_iter:
            break
        
        curr_obj = next_obj
    return Ws
   



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