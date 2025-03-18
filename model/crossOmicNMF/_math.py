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
# (c) 2025 bu1th4nh / UCF Computational Biology Lab. All rights reserved. 
# Written with dedication in the University of Central Florida, EPCOT and the Magic Kingdom.
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
    mlflow_enable:      bool = False
):
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

    if mlflow_enable:
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
        Belle = W @ H @ H.T + alpha * np.sum([degree_block[d][p] @ Wp for p, Wp in enumerate(Ws_current)], axis=0) + betas[d]

        next_W = Ariel / Belle * W
        next_Ws.append(next_W)


    Ariel = np.sum([W.T @ X for W, X in zip(next_Ws, Xs)], axis=0)
    Cindy = np.sum([W.T @ W @ H for W in next_Ws], axis=0) + np.broadcast_to(gammas, (H.shape[0], H.shape[1]))
    next_H = Ariel / Cindy * H

    # # Nullity check
    # self.nullityCheck(H, additional_info=f"H")
    # for d, W in enumerate(next_Ws):
    #     self.nullityCheck(W, additional_info=f"W[{d}]")

    # # Non-negativity check
    # self.negativeCheck(next_H, additional_info=f"H")
    # for d, W in enumerate(next_Ws):
    #     self.negativeCheck(W, additional_info=f"W[{d}]")


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
    curr_obj = objective_function(self.Xd, Ws, H, E, degree_block, self.alpha, self.betas, self.gammas, iteration, self.mlflow_enable)
    if self.mlflow_enable: mlflow.log_metric("objective_function", curr_obj, step=iteration)

    if additional_tasks is not None: 
        if callable(additional_tasks): 
            additional_tasks(Ws, H, iteration)
        else: 
            for task in additional_tasks: task(Ws, H, iteration)

    while True:
        iteration += 1
        new_Ws, new_H = update(self.Xd, Ws, H, E, degree_block, self.alpha, self.betas, self.gammas)

        # Log the delta of Ws and H
        if self.mlflow_enable:
            for W_idx, W in enumerate(new_Ws):
                mlflow.log_metric(f"W{W_idx}_delta", np.linalg.norm(W - Ws[W_idx], 'fro'), step=iteration)
            mlflow.log_metric("H_delta", np.linalg.norm(new_H - H, 'fro'), step=iteration)

        # Update the Ws and H
        Ws = new_Ws
        H = new_H

        # Compute the objective function
        next_obj = objective_function(self.Xd, Ws, H, E, degree_block, self.alpha, self.betas, self.gammas, iteration, self.mlflow_enable)
        delta = next_obj - curr_obj
        logging.info(f"Iteration {iteration}: Objective function = {next_obj}, delta = {delta}")

        # Evaluate metrics if provided
        if additional_tasks is not None and iteration % additional_tasks_interval == 0:
            if callable(additional_tasks): 
                additional_tasks(Ws, H, iteration)
            else: 
                for task in additional_tasks: task(Ws, H, iteration)

        # Log the objective function and delta to MLFlow
        if self.mlflow_enable:
            mlflow.log_metric("objective_function", next_obj, step=iteration)
            mlflow.log_metric("delta", np.abs(delta), step=iteration)

        # Break condition
        if np.abs(delta) < self.tol or iteration >= self.max_iter:
            break
        
        curr_obj = next_obj
    if self.mlflow_enable:
        mlflow.log_metric("Iterations to converge", iteration)

    return Ws, H
   


# Content and code by bu1th4nh/UCF Compbio. Written with dedication in the University of Central Florida and the Magic Kingdom.
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
                                                                                                                                       
