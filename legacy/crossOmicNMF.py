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




def SimilarityMetrics(
    omics_data: np.ndarray, 
    method: Union[Callable, None] = None,
    get_abs_value: bool = True
) -> np.ndarray:
    """
        Calculate the similarity matrix between features in the omics data matrix X (1). The similarity score can further take the absolute value if `get_abs_value` is True
        
        Input
        -----
        `omics_data`: np.ndarray
            The omics data matrix of shape (feature, sample), or (m, N)
        `method`: Union[Callable, None]
            The similarity metric to calculate the similarity matrix. If None, the default metric is Pearson correlation coefficient. The method should take the omics data matrix as input and return the similarity matrix of shape (feature, feature), or (m, m)
        `get_abs_value`: bool
            Whether to take the absolute value of the similarity matrix or not


        Output
        ------
        `similarity_matrix`: np.ndarray
            The similarity matrix of shape (feature, feature), or (m, m)
    """
    
    corr = np.corrcoef(omics_data) if method is None else method(omics_data)

    print(f"Corr matrix shape: {corr.shape}")

    return np.abs(corr) if get_abs_value else corr



def Concatenate(
    on_diagonal_interaction: List[np.ndarray], 
    off_diagonal_interaction: Dict[Tuple[int, int], np.ndarray],
    omics_feature_sizes: List[int]
) -> List[List[np.ndarray]]:
    """
        Concatenate the on-diagonal and off-diagonal interaction matrices (2)

        The reason for installing dict as the off-diagonal interaction is to easily access the interaction matrix of two omics data and can address the non-symmetric interaction case. If there is no interaction between two omics data p and q, the interaction matrix should be the transpose of the interaction matrix between q and p. If neither of the matrices is found, the interaction matrix should be zero matrix
        
        Input
        -----
        `on_diagonal_interaction`: List[np.ndarray]
            A list of on-diagonal interaction matrices of shape (m_d, m_d)
        `off_diagonal_interaction`: Dict[Tuple[int, int], np.ndarray]
            A dictionary of off-diagonal interaction matrices of shape (m_p, m_q) with key as (p, q) where p, q are the interaction matrix of p-th and q-th omics data
        `omics_feature_sizes`: List[int]
            A list of feature sizes of each omics data, i.e. [m_1, m_2, ..., m_D]
        


        Output
        ------
        `similarity_blocks`: List[List[np.ndarray]]
            The concatenated similarity matrix of shape (M, M)
    """
    
    similarity_blocks = []
    # Construct the concatenated similarity matrix
    for i in range(len(omics_feature_sizes)):
        row = []
        for j in range(len(omics_feature_sizes)):
            if i == j:
                row.append(on_diagonal_interaction[i])
            else:
                if (i, j) in off_diagonal_interaction:
                    row.append(off_diagonal_interaction[(i, j)])
                elif (j, i) in off_diagonal_interaction:
                    row.append(off_diagonal_interaction[(j, i)].T)
                else:
                    row.append(np.zeros((omics_feature_sizes[i], omics_feature_sizes[j])))
        similarity_blocks.append(row)

    
    # Debug: Output the concatenated similarity matrix shape
    print(f"Concatenated similarity matrix shape: {np.block(similarity_blocks).shape}")
    for h_block in similarity_blocks:
        for v_block in h_block:
            print(f"Block shape: {v_block.shape}")

                

    # Double check the concatenated similarity matrix    
    # Check if each blocks on the same row have the same number of rows
    assert all([len(set([block.shape[0] for block in row])) == 1 for row in similarity_blocks]), "Each block on the same row should have the same number of rows"

    # Check if each blocks on the same column have the same number of columns
    col_shapes = np.array([[block.shape[1] for block in row] for row in similarity_blocks]).T
    assert all([len(set(col)) == 1 for col in col_shapes]), "Each block on the same column should have the same number of columns"

    return similarity_blocks



def Normalize(
        similarity_block: List[List[np.ndarray]],

        off_diag_method: Union[Callable, Literal["l2", "l1", "max", "min", "mean", "median", "sum", "passthru"]] = "passthru",
        on_diag_method: Union[Callable, Literal["l2", "l1", "max", "min", "mean", "median", "sum", "passthru"]] = "passthru",
        whole_block_method: Union[Callable, Literal["l2", "l1", "max", "min", "mean", "median", "sum", "passthru"]] = "passthru",
        
        off_diag_norm_orientation: Literal["row", "column", "all"] = "row",
        on_diag_norm_orientation: Literal["row", "column", "all"] = "row",
        whole_block_norm_orientation: Literal["row", "column", "all"] = "row"
    ) -> List[List[np.ndarray]]:
    """
        Normalize the similarity matrix (3)

        By isolating the normalization step, we can easily replace the normalization method with other methods for each and all blocks
        
        Input
        -----
        `similarity_block`: List[List[np.ndarray]]
            The concatenated similarity matrix of shape (M, M)

        `off_diag_method`, `on_diag_norm_orientation`, `whole_block_norm_orientation`: Union[Callable, Literal["l2", "l1", "max", "min", "mean", "median", "sum", "passthru"]]
            The normalization method for off-diagonal, on-diagonal, and whole block. If a callable is passed, the method will be used directly. If a string is passed, the method will be selected from the predefined methods. If "passthru" is passed, the matrix will be passed through without any normalization

        `off_diag_norm_orientation`, `on_diag_norm_orientation`, `whole_block_norm_orientation`: Literal["row", "column", "all"]
            The orientation to normalize the matrix. If "row", the matrix will be normalized along the row axis. If "column", the matrix will be normalized along the column axis. If "all", the matrix will be normalized as a whole. When "passthru" is passed, this parameter will be ignored 



        Output
        ------
        `normalized_similarity_matrix`: List[List[np.ndarray]]
            The normalized similarity matrix of shape (M, M)
    """

    def divisor_func_selection(
        method: Literal["l2", "l1", "max", "min", "mean", "median", "sum"],
        axis: Literal["row", "column", "all"]
    ) -> Callable:
        
        if axis == "row": orientation, shape = 1, (-1, 1)
        elif axis == "column": orientation, shape = 0, (1, -1)
        else: orientation, shape = None, -1

        # print(orientation, shape)

        if method == "l2":       return lambda x: np.reshape(np.linalg.norm(x, ord=2, axis=orientation), shape = shape)
        elif method == "l1":     return lambda x: np.reshape(np.linalg.norm(x, ord=1, axis=orientation), shape = shape)
        elif method == "max":    return lambda x: np.reshape(np.max(x, axis=orientation), shape = shape)
        elif method == "min":    return lambda x: np.reshape(np.min(x, axis=orientation), shape = shape)
        elif method == "mean":   return lambda x: np.reshape(np.mean(x, axis=orientation), shape = shape)
        elif method == "median": return lambda x: np.reshape(np.median(x, axis=orientation), shape = shape)
        elif method == "sum":    return lambda x: np.reshape(np.sum(x, axis=orientation), shape = shape)
        else: return None

    
    def norm_func_selection(
        method: Union[Callable, Literal["l2", "l1", "max", "min", "mean", "median", "sum", "passthru"]],
        orientation: Literal["row", "column", "all"]
    ) -> Callable:
        if method == "passthru": return lambda x: x
        elif callable(method): return method
        else: return divisor_func_selection(method, orientation)


    # Component-wise normalization
    ## Norm function selection
    on_diag_norm = norm_func_selection(on_diag_method, on_diag_norm_orientation)
    off_diag_norm = norm_func_selection(off_diag_method, off_diag_norm_orientation)
    ## Normalize each block
    for i in range(len(similarity_block)):
        for j in range(len(similarity_block)):
            norm = on_diag_norm if i == j else off_diag_norm
            similarity_block[i][j] /= (norm(similarity_block[i][j]) + 1)

    # raise Exception("Stop here")


    # Whole-block normalization
    ## Normalize the whole block
    whole_block_norm = norm_func_selection(whole_block_method, whole_block_norm_orientation)
    tmp_block = np.block(similarity_block)
    tmp_block /= (whole_block_norm(tmp_block) + 1)
    ## Derive omics size
    omics_sizes = [block.shape[0] for row in similarity_block for block in row]
    ## Derive cut indices. 
    omics_indices = np.cumsum(omics_sizes)[:-1] # Drop the final index
    ## Break the block matrix back to the list of list of matrices
    similarity_block = [
        list(np.hsplit(horizontal_block, omics_indices))
        for horizontal_block 
        in np.vsplit(tmp_block, omics_indices)
    ]

    print(f"Normalized similarity shape after whole block: {np.block(similarity_block)}")

    return similarity_block



def InitializeWd(omics_data: List[np.ndarray], num_clusters: int) -> List[np.ndarray]:
    """
        Initialize the W matrices for each omic data (4)
        
        Input
        -----
        `omics_data`: List[np.ndarray]
            A list of omics data matrices of shape (m_d, N)
        `num_clusters`: int
            The number of clusters to initialize the W matrices
        


        Output
        ------
        W: List[np.ndarray]
            A list of W matrices of shape (m_d, k)
    """

    # Now only use random
    return [np.random.rand(omic.shape[0], num_clusters) for omic in omics_data]
    


def LassoSolveH(
    omics_data: List[np.ndarray], 
    initialized_Wds: List[np.ndarray], 
    lasso_control_lambdas: Union[np.array, List, float]
) -> np.ndarray:
    """
        Solve the H matrix with fixed W matrices (5)
        
        Input
        -----
        `omics_data`: List[np.ndarray]
            A list of omics data matrices of shape (m_d, N)
        `initialized_Wds`: List[np.ndarray]
            A list of initialized W matrices of shape (m_d, k). The cluster size will be self-inferred from the shape of the W matrices
        `lasso_control_lambdas`: Union[np.array, List]
            The L1 regularization parameter for each sample. If a single value is passed, the same value will be used for all samples. If a list is passed, the value will be used for each sample
        


        Output
        ------
        H: np.ndarray
            The H matrix of shape (k, N)
    """

    # Basic sizes
    # D = len(omics_data)
    N = omics_data[0].shape[1]
    # m = [omic.shape[0] for omic in omics_data]

    # Concat all omics data matrix along the column axis, aka big_X = [X_1; X_2; ...; X_D] with shape (m_1 + m_2 + ... + m_D, N)
    big_X = np.concatenate(omics_data, axis=0)
    print(f"Big X shape: {big_X.shape}")
    
    # Concat all Ws data matrix along the column axis, aka big_W = [W_1; W_2; ...; W_D] with shape (m_1 + m_2 + ... + m_D, k)
    big_W = np.concatenate(initialized_Wds, axis=0)
    print(f"Big W shape: {big_W.shape}")

    # Initialize H
    H_coeffs = []

    # Solve individual column of H
    alphas = np.array([lasso_control_lambdas] * N) if isinstance(lasso_control_lambdas, (int, float)) else lasso_control_lambdas
    for index, alpha in enumerate(alphas):
        lasso = Lasso(alpha = alpha, fit_intercept = False);
        lasso.fit(
            big_W,
            big_X[:, index],
        )
        H_coeffs.append(np.array(lasso.coef_.T))

    # Construct H
    return np.array(H_coeffs).T



def IterativeSolveWds(
    omics_data:             List[np.ndarray], 
    initialized_Wds:        List[np.ndarray], 
    H:                      np.ndarray, 
    cross_omic_similarity:  List[List[np.ndarray]],

    cross_omics_alpha:      float,
    sparsity_control_betas: Union[float, List[float]],
    lasso_control_lambdas:   Union[np.array, List, float],

    max_iter: int = 100,
    tol: float = 1e-8
) -> List[np.ndarray]:
    """
        Iteratively solve the W matrices with fixed H matrix (6)
        
        Input
        -----
        `omics_data`: List[np.ndarray]
            A list of omics data matrices of shape (m_d, N)
        `initialized_Wds`: List[np.ndarray]
            A list of initialized W matrices of shape (m_d, k)
        `H`: np.ndarray
            The H matrix of shape (k, N)
        `cross_omic_similarity`: List[List[np.ndarray]]
            The concatenated similarity matrix block of shape (M, M)

        `cross_omics_alpha`: float
            The hyperparameter to control the graph regularization term
        `sparsity_control_betas`: Union[float, List[float]]
            The hyperparameter to control the sparsity of W matrices. If a single value is passed, the same value will be used for all W matrices. If a list is passed, the value will be used for each W matrix
        `lasso_control_lambdas`: Union[np.array, List]
            The L1 regularization parameter for each sample. If a single value is passed, the same value will be used for all samples. If a list is passed, the value will be used for each sample

        `max_iter`: int
            The maximum number of iterations to run the algorithm
        `tol`: float
            The tolerance to stop the algorithm
        


        Output
        ------
        W: List[np.ndarray]
            A list of W matrices of shape (m_d, k)
    """
    
    def objective_function(
        Xs:                 List[np.ndarray], 
        Ws:                 List[np.ndarray], 
        H:                  np.ndarray, 
        similarity_block:   List[List[np.ndarray]], 
        degree_block:       List[List[np.ndarray]], 
        
        alpha:              float, 
        betas:              Union[np.ndarray, List[float]],
        lambdas:            Union[np.ndarray, List[float]]
    ):
        X_concatted = np.concatenate(Xs, axis=0)
        W_concatted = np.concatenate(Ws, axis=0)


        print(f"Xs: {Xs}")
        print(f"Ws: {Ws}")
        print(f"H: {H}")
        print(f"Sim: {np.block(similarity_block)}")
        print(f"Deg: {np.block(degree_block)}")



        # Calculate the reconstruction error
        reconstruction_error = np.linalg.norm(X_concatted - W_concatted @ H, ord="fro") ** 2

        # Graph regularization term
        laplacian_block = np.block(degree_block) - np.block(similarity_block)
        graph_regularization = alpha * np.trace(W_concatted.T @ laplacian_block @ W_concatted)

        # Sparsity control term for W
        sparsity_control_for_Ws = np.array([np.linalg.norm(W, ord=1) for W in Ws]) @ np.array(betas)

        # Sparsity control term for H
        sparsity_control_for_H = np.linalg.norm(H, ord=1, axis = 0) @ np.array(lambdas)
        return reconstruction_error + graph_regularization + sparsity_control_for_Ws + sparsity_control_for_H
    

    def update_Ws(Xs, Ws_current, H, similarity_block, degree_block, alpha, betas):
        next_Ws = []
        for d, W in enumerate(Ws_current):
            Alpha = W @ H @ H.T + alpha * np.sum([similarity_block[d][p] @ Wp for p, Wp in enumerate(Ws_current)], axis=0)
            Bravo = Xs[d] @ H.T + alpha * np.sum([degree_block[d][p] @ Wp for p, Wp in enumerate(Ws_current)], axis=0) + betas[d] * np.sign(W)

            next_W = Alpha / Bravo * W
            next_Ws.append(next_W)
        return next_Ws



    # Basic sizes
    D = len(omics_data)
    N = omics_data[0].shape[1]
    m = [omic.shape[0] for omic in omics_data]

    # Hyperparameters
    sparsity_control_betas = np.array([sparsity_control_betas] * D) if isinstance(sparsity_control_betas, (int, float)) else sparsity_control_betas
    lasso_control_lambdas = np.array([lasso_control_lambdas] * N) if isinstance(lasso_control_lambdas, (int, float)) else lasso_control_lambdas

    # Construct the degree matrix
    omics_indices = np.cumsum(m)[:-1] # Drop the final index
    degree_block = [
        list(np.hsplit(horizontal_block, omics_indices))
        for horizontal_block 
        in np.vsplit(np.eye(np.sum(m)), omics_indices)
    ]

    print(f"Degree block shape: {omics_indices}")
    for hblk in degree_block:
        for vblk in hblk:
            print(f"Block shape: {vblk.shape}")

    print(f"Degree block shape: {np.block(degree_block).shape}")
    print(f"Similarity block shape: {np.block(cross_omic_similarity).shape}")
    
    # Iteratively solve the W matrices
    Ws = initialized_Wds
    iteration = 0
    curr_obj = objective_function(omics_data, Ws, H, cross_omic_similarity, degree_block, cross_omics_alpha, sparsity_control_betas, lasso_control_lambdas)

    print(omics_data)

    while True:
        Ws = update_Ws(omics_data, Ws, H, cross_omic_similarity, degree_block, cross_omics_alpha, sparsity_control_betas)
        next_obj = objective_function(omics_data, Ws, H, cross_omic_similarity, degree_block, cross_omics_alpha, sparsity_control_betas, lasso_control_lambdas)

        iteration += 1
        print(f"Iteration {iteration}: Objective function = {next_obj}")

        if np.abs(next_obj - curr_obj) < tol or iteration >= max_iter:
            break
        
        curr_obj = next_obj
    return Ws



def mainCrossOmics(
    omics_data:                 List[np.ndarray],
    off_diagonal_interaction:   Dict[Tuple[int, int], np.ndarray],
    

    num_clusters: int,
    lasso_control_lambdas:      Union[np.array, List, float],
    cross_omics_alpha:          float,
    sparsity_control_betas:     Union[float, List[float]],


    max_iter: int, 
    tol: float
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
        Main function to run the CrossOmicDataInt algorithm
        
        Input
        -----
        `omics_data`: List[np.ndarray]
            A list of omics data matrices of shape (m_d, N)
        `num_clusters`: int
            The number of clusters to initialize the W matrices
        `max_iter`: int
            The maximum number of iterations to run the algorithm
        `tol`: float
            The tolerance to stop the algorithm
        


        Output
        ------
        W: List[np.ndarray]
            A list of W matrices of shape (m_d, k)
        H: np.ndarray
            The H matrix of shape (k, N)
    """

    # Initialization
    # Omics size
    D = len(omics_data)
    # Feature size
    m = [omic.shape[0] for omic in omics_data]

    # Calculate the similarity matrix for within-omic data
    print('Calculating the similarity matrix for within-omic data')
    on_diagonal_similarity = [SimilarityMetrics(omic) for omic in omics_data]

    # Concatenate the similarity matrix
    print('Concatenating the similarity matrix')
    concatenated_similarity = Concatenate(on_diagonal_similarity, off_diagonal_interaction, m)

    # Normalize the concatenated similarity matrix
    print('Normalizing the concatenated similarity matrix')
    normalized_similarity = Normalize(
        concatenated_similarity,
        off_diag_method = "sum",
        on_diag_method = "sum",
        whole_block_method = "sum",
        off_diag_norm_orientation = "row",
        on_diag_norm_orientation = "row",
        whole_block_norm_orientation = "row"
    )

    print(f"Normalized similarity shape: {np.block(normalized_similarity).shape}")

    # Initialize Wds matrices
    print('Initializing W matrices')
    initialized_Wds = InitializeWd(omics_data, num_clusters)

    # Solve the H matrix
    print('Solving the H matrix using Lasso')
    H = LassoSolveH(
        omics_data, 
        initialized_Wds,
        lasso_control_lambdas
    )
    print(f"H shape: {H.shape}")

    # Iteratively solve the W matrices
    print(f"Normalized similarity shape before solving W: {np.block(normalized_similarity).shape}")
    print('Iteratively solving the W matrices')
    Wds = IterativeSolveWds(
        # Data
        omics_data, 
        initialized_Wds, 
        H, 
        normalized_similarity, 

        # Hyperparameters
        cross_omics_alpha, 
        sparsity_control_betas, 
        lasso_control_lambdas, 
        
        # Control parameters
        max_iter, 
        tol
    )
    return Wds, H





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
                                                                                                                                       
