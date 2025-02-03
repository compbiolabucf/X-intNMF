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
import logging
import mlflow
import torch
from torch import Tensor



# @jit(forceobj=True)
def torch_objective_function(
    # self,

    Xs:                 List[Tensor], 
    Ws:                 List[Tensor], 
    H:                  Tensor, 
    similarity_block:   List[List[Tensor]], 
    degree_block:       List[List[Tensor]], 
    
    alpha:              float, 
    betas:              Tensor,
    gammas:             Tensor,

    iteration:          int,
    device:             str,
) -> np.float64:

    # Calculate the reconstruction error
    reconstruction_error = 0 
    for p in range(len(Ws)):
        reconstruction_error += torch.norm(Xs[p] - Ws[p] @ H, p="fro") ** 2
    reconstruction_error = reconstruction_error.detach().cpu().numpy()


    # Graph regularization term
    # laplacian_block = np.block(degree_block) - np.block(similarity_block)
    # graph_regularization = alpha/2 * np.trace(W_concatted.T @ laplacian_block @ W_concatted)
    graph_regularization = 0.0
    for p in range(len(Ws)):
        for q in range(len(Ws)):
            graph_regularization += alpha/2 * torch.trace(Ws[p].T @ (degree_block[p][q] - similarity_block[p][q]) @ Ws[q]).detach().cpu().numpy()
            



    # Sparsity control term for W
    sparsity_control_for_Ws = Tensor([torch.norm(W, p=1) for W in Ws]).to(device) @ betas
    sparsity_control_for_Ws = sparsity_control_for_Ws.detach().cpu().numpy()


    # Sparsity control term for H
    sparsity_control_for_H = torch.norm(H, p=1, dim=0) @ gammas
    sparsity_control_for_H = sparsity_control_for_H.detach().cpu().numpy()
    

    # Objective function
    f = 1/2 * reconstruction_error + graph_regularization + sparsity_control_for_Ws + sparsity_control_for_H



    mlflow.log_metric("Reconstruction error", reconstruction_error, step=iteration)
    mlflow.log_metric("Graph regularization", graph_regularization, step=iteration)
    mlflow.log_metric("Sparsity control for Ws", sparsity_control_for_Ws, step=iteration)
    mlflow.log_metric("Sparsity control for H", sparsity_control_for_H, step=iteration)

    return f



# @jit(forceobj=True)
def torch_update(
    # self,

    Xs:                 List[Tensor], 
    Ws_current:         List[Tensor], 
    H:                  Tensor, 
    similarity_block:   List[List[Tensor]], 
    degree_block:       List[List[Tensor]], 

    alpha:              float, 
    betas:              Union[Tensor, List[float]],
    gammas:             Union[Tensor, List[float]],
) -> Tuple[List[Tensor], Tensor]:
    
    next_Ws = []
    for d, W in enumerate(Ws_current):
        Ariel = Xs[d] @ H.T + alpha * torch.sum(torch.stack([similarity_block[d][p] @ Wp for p, Wp in enumerate(Ws_current)]), dim=0)
        Belle = W @ H @ H.T + alpha * torch.sum(torch.stack([degree_block[d][p] @ Wp for p, Wp in enumerate(Ws_current)]), dim=0) + betas[d]

        next_W = Ariel / Belle * W
        next_Ws.append(next_W)


    Ariel = torch.sum(torch.stack([W.T @ X for W, X in zip(next_Ws, Xs)]), dim=0)
    Cindy = torch.sum(torch.stack([W.T @ W @ H for W in next_Ws]), dim=0) + torch.broadcast_to(gammas, (H.shape[0], H.shape[1]))
    next_H = Ariel / Cindy * H
    


    return next_Ws, next_H




def IterativeSolveWdsAndH_PyTorch(
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

    # Construct the omic indices for matrix splitting
    omics_indices = np.cumsum(self.m)[:-1] # Drop the final index
    

    
    # Construct the normalized similarity matrix
    E = np.block(self.E)
    D = np.diag(1 / np.sqrt(np.sum(E, axis=1)))

    # Pytorch Extension
    E_tensor = Tensor(E).to(self.device)
    D_tensor = Tensor(D).to(self.device)
    E_normalized_tensor = D_tensor @ E_tensor @ D_tensor
    E_normalized = E_normalized_tensor.detach().cpu().numpy()

    # Convert the similarity block to a list of matrices
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




    # PyTorch Extension
    initialized_Wds = [Tensor(W).to(self.device) for W in initialized_Wds]
    initialized_H = Tensor(initialized_H).to(self.device)
    for p in range(self.D):
        for q in range(self.D):
            E[p][q] = Tensor(E[p][q]).to(self.device)
            degree_block[p][q] = Tensor(degree_block[p][q]).to(self.device)
    alpha = self.alpha
    betas = Tensor(self.betas).to(self.device)
    gammas = Tensor(self.gammas).to(self.device)
    Xd = [Tensor(X).to(self.device) for X in self.Xd]


    
    # Iteratively solve the W matrices
    Ws = initialized_Wds
    H = initialized_H
    iteration = 0
    curr_obj = torch_objective_function(Xd, Ws, H, E, degree_block, alpha, betas, gammas, iteration, self.device)
    mlflow.log_metric("objective_function", curr_obj, step=iteration)

    if additional_tasks is not None: 
        if callable(additional_tasks): 
            additional_tasks(Ws, H, iteration)
        else: 
            for task in additional_tasks: task(Ws, H, iteration)


    while True:
        iteration += 1
        new_Ws, new_H = torch_update(Xd, Ws, H, E, degree_block, alpha, betas, gammas)

        # Log the delta of Ws and H
        for W_idx, W in enumerate(new_Ws):
            mlflow.log_metric(f"W{W_idx}_delta", torch.norm(new_Ws[W_idx] - Ws[W_idx], p="fro").detach().cpu().numpy(), step=iteration)
        mlflow.log_metric(f"H_delta", torch.norm(new_H - H, p="fro").detach().cpu().numpy(), step=iteration)

        # Update the Ws and H
        Ws = new_Ws
        H = new_H

        # Compute the objective function
        next_obj = torch_objective_function(Xd, Ws, H, E, degree_block, alpha, betas, gammas, iteration, self.device)
        delta = next_obj - curr_obj
        logging.info(f"Iteration {iteration}: Objective function = {next_obj}, delta = {delta}")

        # Evaluate metrics if provided
        if additional_tasks is not None and iteration % additional_tasks_interval == 0:
            if callable(additional_tasks): 
                additional_tasks(Ws, H, iteration)
            else: 
                for task in additional_tasks: task(Ws, H, iteration)


        # Log the objective function and delta to MLFlow
        mlflow.log_metric("objective_function", next_obj, step=iteration)
        mlflow.log_metric("delta", np.abs(delta), step=iteration)

        # Break condition
        if np.abs(delta) < self.tol or iteration >= self.max_iter:
            break
        
        curr_obj = next_obj

    mlflow.log_metric("Iterations to converge", iteration)

    Ws = [W.detach().cpu().numpy() for W in Ws]
    H = H.detach().cpu().numpy()
    return Ws, H