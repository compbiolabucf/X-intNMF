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
# Description: File contains the objective and iterative functions with default NumPy backend
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
import numpy as np
import pandas as pd
import logging
import mlflow




# @jit(forceobj=True)
def objective_function(
    # self,

    Xs:                 List[np.ndarray], 
    Ws:                 List[np.ndarray], 
    H:                  np.ndarray, 
    similarity_block:   List[List[np.ndarray]], 
    degree_block:       List[List[np.ndarray]], 
    
    alpha:              float, 
    betas:              Union[np.ndarray, List[float]],
    gammas:             Union[np.ndarray, List[float]],

    iteration:          int,
    mlflow_enabled:     bool = True,
) -> np.float64:
    X_concatted = np.concatenate(Xs, axis=0)
    W_concatted = np.concatenate(Ws, axis=0)


    # print(f"Xs: {Xs}")
    # print(f"Ws: {Ws}")
    # print(f"H: {H}")
    # print(f"Sim: {np.block(similarity_block)}")
    # print(f"Deg: {np.block(degree_block)}")

    
    # Nullity check
    # self.nullityCheck(X_concatted, additional_info="X_concatted")
    # self.nullityCheck(W_concatted, additional_info="W_concatted")
    # self.nullityCheck(H, additional_info="H")
    # self.nullityCheck(np.block(similarity_block), additional_info="similarity_block")
    # self.nullityCheck(np.block(degree_block), additional_info="degree_block")

    # Non-negativity check
    # self.negativeCheck(X_concatted, additional_info="X_concatted")
    # self.negativeCheck(W_concatted, additional_info="W_concatted")
    # self.negativeCheck(H, additional_info="H")
    # self.negativeCheck(np.block(similarity_block), additional_info="similarity_block")
    # self.negativeCheck(np.block(degree_block), additional_info="degree_block")


    # Calculate the reconstruction error
    reconstruction_error = np.linalg.norm(X_concatted - W_concatted @ H, ord="fro") ** 2

    # Graph regularization term
    laplacian_block = np.block(degree_block) - np.block(similarity_block)
    graph_regularization = alpha/2 * np.trace(W_concatted.T @ laplacian_block @ W_concatted)

    # Sparsity control term for W
    sparsity_control_for_Ws = np.array([np.linalg.norm(W, ord=1) for W in Ws]) @ np.array(betas)

    # Sparsity control term for H
    sparsity_control_for_H = np.linalg.norm(H, ord=1, axis = 0) @ np.array(gammas)
    f = 1/2 * reconstruction_error + graph_regularization + sparsity_control_for_Ws + sparsity_control_for_H

    # logging.info(f"Reconstruction error: {reconstruction_error}")
    # logging.info(f"Graph regularization: {graph_regularization}")
    # logging.info(f"Sparsity control for Ws: {sparsity_control_for_Ws}")
    # logging.info(f"Sparsity control for H: {sparsity_control_for_H}")
    # logging.info(f"Objective function value: {f}")

    if mlflow_enabled:
        mlflow.log_metric("Reconstruction error", reconstruction_error, step=iteration)
        mlflow.log_metric("Graph regularization", graph_regularization, step=iteration)
        mlflow.log_metric("Sparsity control for Ws", sparsity_control_for_Ws, step=iteration)
        mlflow.log_metric("Sparsity control for H", sparsity_control_for_H, step=iteration)

    return f



# @jit(forceobj=True)
def update(
    # self,

    Xs:                 List[np.ndarray], 
    Ws_current:         List[np.ndarray], 
    H:                  np.ndarray, 
    similarity_block:   List[List[np.ndarray]], 
    degree_block:       List[List[np.ndarray]], 

    alpha:              float, 
    betas:              Union[np.ndarray, List[float]],
    gammas:             Union[np.ndarray, List[float]]
) -> Tuple[List[np.ndarray], np.ndarray]:
    
    next_Ws = []
    for d, W in enumerate(Ws_current):
        Ariel = Xs[d] @ H.T + alpha * np.sum([similarity_block[d][p] @ Wp for p, Wp in enumerate(Ws_current)], axis=0)
        Belle = W @ H @ H.T + alpha * W + betas[d] + 0.0001 # Add 0.0001 to avoid division by zero
        next_W = Ariel / Belle * W


        # Normalize the next_W, to have sum of each row equal to 1
        # next_W = next_W / np.sum(next_W, axis=1, keepdims=True)
        # Clip the next_W to be non-negative
        # next_W = np.clip(next_W, 0, None)
        # Fill NaN/Inf with 0
        # next_W = np.nan_to_num(next_W, nan=0, posinf=0, neginf=0)
        
        next_Ws.append(next_W)


    Ariel = np.sum([W.T @ X for W, X in zip(next_Ws, Xs)], axis=0)
    Cindy = np.sum([W.T @ W @ H for W in next_Ws], axis=0) + np.broadcast_to(gammas, (H.shape[0], H.shape[1]))
    next_H = Ariel / Cindy * H

    # Normalize the next_H, to have sum of each columns equal to 1
    # next_H = next_H / np.sum(next_H, axis=0, keepdims=True)
    # Clip the next_H to be non-negative
    # next_H = np.clip(next_H, 0, None)




    return next_Ws, next_H





def IterativeSolveWdsAndH(
    self,
    initialized_Wds:            List[np.ndarray], 
    initialized_H:              np.ndarray, 
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
    H = initialized_H
    iteration = 0
    curr_obj = objective_function(self.Xd, Ws, H, E, degree_block, self.alpha, self.betas, self.gammas, iteration, self.mlflow_enabled)
    if self.mlflow_enabled: mlflow.log_metric("objective_function", curr_obj, step=iteration)


    if additional_tasks is not None: 
        if callable(additional_tasks): additional_tasks(Ws, H, iteration)
        else: _ = [task(Ws, H, iteration) for task in additional_tasks] # Execute the additional tasks


    while True:
        iteration += 1
        new_Ws, new_H = update(self.Xd, Ws, H, E, degree_block, self.alpha, self.betas, self.gammas)

        # Log the delta of Ws and H
        if self.mlflow_enabled:
            for W_idx, W in enumerate(new_Ws):
                mlflow.log_metric(f"W{W_idx}_delta", np.linalg.norm(W - Ws[W_idx], 'fro'), step=iteration)
            mlflow.log_metric("H_delta", np.linalg.norm(new_H - H, 'fro'), step=iteration)


        # Update the Ws and H
        Ws = new_Ws
        H = new_H


        # Compute the objective function
        next_obj = objective_function(self.Xd, Ws, H, E, degree_block, self.alpha, self.betas, self.gammas, iteration, self.mlflow_enabled)
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
            mlflow.log_metric("delta", np.abs(delta), step=iteration)


        # Break condition
        if np.abs(delta) < self.tol or iteration >= self.max_iter:
            if np.abs(delta) < self.tol: logging.info(f"Converged!")
            break
        curr_obj = next_obj
        

    logging.info(f"Finished after {iteration} iterations.")
    if self.mlflow_enabled: mlflow.log_metric("Iterations to converge", iteration)

    return Ws, H
   
