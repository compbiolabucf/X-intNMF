# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành (@bu1th4nh)
# Title: _math.py
# Date: 2024/09/22 13:42:21
# Description: File contains the objective and iterative functions
# 
# (c) bu1th4nh. All rights reserved
# -----------------------------------------------------------------------------------------------


from typing import List, Tuple, Union, Literal, Any, Callable, Dict
import pandas as pd
import numpy as np
import cupy as cp
import logging
import mlflow




# @jit(forceobj=True)
def cupy_objective_function(
    # self,

    Xs:                 List[cp.ndarray], 
    Ws:                 List[cp.ndarray], 
    H:                  cp.ndarray, 
    similarity_block:   List[List[cp.ndarray]], 
    degree_block:       List[List[cp.ndarray]], 
    
    alpha:              float, 
    betas:              Union[cp.ndarray, List[float]],
    gammas:             Union[cp.ndarray, List[float]],

    iteration:          int,
    mlflow_enabled:     bool = True,
) -> np.float64:

    # Calculate the reconstruction error
    reconstruction_error = 0 #cp.sum(cp.array([cp.linalg.norm(X - W @ H, ord="fro") ** 2 for X, W in zip(Xs, Ws)])) # np.linalg.norm(X_concatted - W_concatted @ H, ord="fro") ** 2
    for p in range(len(Ws)):
        reconstruction_error += cp.linalg.norm(Xs[p] - Ws[p] @ H, ord="fro") ** 2


    # Graph regularization term
    # laplacian_block = np.block(degree_block) - np.block(similarity_block)
    # graph_regularization = alpha/2 * np.trace(W_concatted.T @ laplacian_block @ W_concatted)
    graph_regularization = cp.float64(0.0)
    for p in range(len(Ws)):
        for q in range(len(Ws)):
            graph_regularization += alpha/2 * cp.trace(Ws[p].T @ (degree_block[p][q] - similarity_block[p][q]) @ Ws[q])



    # Sparsity control term for W
    sparsity_control_for_Ws = cp.array([cp.linalg.norm(W, ord=1) for W in Ws]) @ cp.array(betas)

    # Sparsity control term for H
    sparsity_control_for_H = cp.linalg.norm(H, ord=1, axis = 0) @ cp.array(gammas)
    f = 1/2 * reconstruction_error + graph_regularization + sparsity_control_for_Ws + sparsity_control_for_H


    if mlflow_enabled:
        mlflow.log_metric("Reconstruction error", reconstruction_error, step=iteration)
        mlflow.log_metric("Graph regularization", graph_regularization, step=iteration)
        mlflow.log_metric("Sparsity control for Ws", sparsity_control_for_Ws, step=iteration)
        mlflow.log_metric("Sparsity control for H", sparsity_control_for_H, step=iteration)

    return f if isinstance(f, np.float64) else f.get()



# @jit(forceobj=True)
def cupy_update(
    # self,

    Xs:                 List[cp.ndarray], 
    Ws_current:         List[cp.ndarray], 
    H:                  cp.ndarray, 
    similarity_block:   List[List[cp.ndarray]], 
    degree_block:       List[List[cp.ndarray]], 

    alpha:              float, 
    betas:              Union[cp.ndarray, List[float]],
    gammas:             Union[cp.ndarray, List[float]]
) -> Tuple[List[cp.ndarray], cp.ndarray]:
    
    next_Ws = []
    for d, W in enumerate(Ws_current):
        Ariel = Xs[d] @ H.T + alpha * cp.sum(cp.array([similarity_block[d][p] @ Wp for p, Wp in enumerate(Ws_current)]), axis=0) 
        Belle = W @ H @ H.T + alpha * W + betas[d] + 0.0001 # Add 0.0001 to avoid division by zero
        next_W = Ariel / Belle * W


        next_Ws.append(next_W)

    Ariel = cp.sum(cp.array([W.T @ X for W, X in zip(next_Ws, Xs)]), axis=0)
    Cindy = cp.sum(cp.array([W.T @ W @ H for W in next_Ws]), axis=0) + cp.broadcast_to(gammas, (H.shape[0], H.shape[1]))
    next_H = Ariel / Cindy * H

    
    # Normalize the next_H, to have sum of each columns equal to 1
    # next_H = next_H / cp.sum(next_H, axis=0, keepdims=True)
    # Clip the values to be non-negative
    # next_H = cp.clip(next_H, 0, None)

    


    return next_Ws, next_H





def IterativeSolveWdsAndH_CuPy(
    self,
    initialized_Wds:            Union[List[np.ndarray]], 
    initialized_H:              Union[np.ndarray], 
    additional_tasks:           Union[None, Callable, List[Callable]] = None,
    additional_tasks_interval:  int = 50,
) -> List[np.ndarray]:
    """
        Iteratively solve the W matrices with fixed H matrix (6)
        
        Input
        -----
        `initialized_Wds`: List[np.ndarray]
            A list of initialized W matrices of shape (m_d, k)
        `initialized_H`: np.ndarray
            The initialized H matrix of shape (k, N)
        `additional_tasks`: Callable, optional
            A function to execute the metrics during the iteration. The function should take the current W matrices and H matrix as input and perform the additional tasks with every `additional_tasks_interval` iterations.
        `additional_tasks_interval`: int
            The interval to execute the additional tasks. Default is 50

        Output
        ------
        W: List[np.ndarray]
            A list of W matrices of shape (m_d, k)
    """
    import cupy as cp
    

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


    # CuPy Extension
    initialized_Wds = [cp.asarray(W) for W in initialized_Wds]
    initialized_H = cp.asarray(initialized_H)
    for p in range(self.D):
        for q in range(self.D):
            E[p][q] = cp.asarray(E[p][q])
            degree_block[p][q] = cp.asarray(degree_block[p][q])
    alpha = cp.float64(self.alpha)
    betas = cp.asarray(self.betas)
    gammas = cp.asarray(self.gammas)
    Xd = [cp.asarray(X) for X in self.Xd]



    
    # Iteratively solve the W matrices
    Ws = initialized_Wds
    H = initialized_H
    iteration = 0
    curr_obj = cupy_objective_function(Xd, Ws, H, E, degree_block, alpha, betas, gammas, iteration, self.mlflow_enabled)
    if self.mlflow_enabled: mlflow.log_metric("objective_function", curr_obj, step=iteration)


    if additional_tasks is not None: 
        if callable(additional_tasks): additional_tasks(Ws, H, iteration)
        else: _ = [task(Ws, H, iteration) for task in additional_tasks] # Execute the additional tasks


    while True:
        iteration += 1
        new_Ws, new_H = cupy_update(Xd, Ws, H, E, degree_block, alpha, betas, gammas)


        # Log the delta of Ws and H
        if self.mlflow_enabled:
            for W_idx, W in enumerate(new_Ws):
                mlflow.log_metric(f"W{W_idx}_delta", cp.linalg.norm(W - Ws[W_idx], 'fro'), step=iteration)
            mlflow.log_metric("H_delta", cp.linalg.norm(new_H - H, 'fro'), step=iteration)


        # Update the Ws and H
        Ws = new_Ws
        H = new_H


        # Compute the objective function
        next_obj = cupy_objective_function(Xd, Ws, H, E, degree_block, alpha, betas, gammas, iteration, self.mlflow_enabled)
        delta = next_obj - curr_obj
        logging.info(f"Iteration {iteration}: Objective function = {next_obj}, delta = {delta}")
        if np.isnan(next_obj) or np.isinf(next_obj):
            logging.error(f"Objective function is NaN/Inf at iteration {iteration}.")
            break


        # Evaluate metrics if provided
        if additional_tasks is not None and iteration % additional_tasks_interval == 0:
            if callable(additional_tasks): additional_tasks(Ws, H, iteration)
            else: _ = [task(Ws, H, iteration) for task in additional_tasks] # Execute the additional tasks


        # Log the objective function and delta to MLFlow
        if self.mlflow_enabled:
            mlflow.log_metric("objective_function", next_obj, step=iteration)
            mlflow.log_metric("delta", cp.abs(delta), step=iteration)


        # Break condition
        if cp.abs(delta) < self.tol or iteration >= self.max_iter:
            if cp.abs(delta) < self.tol: logging.info(f"Converged!")
            break
        curr_obj = next_obj


    logging.info(f"Finished after {iteration} iterations.")
    if self.mlflow_enabled: mlflow.log_metric("Iterations to converge", iteration)

    Ws = [W.get() for W in Ws]
    H = H.get()
    return Ws, H
   