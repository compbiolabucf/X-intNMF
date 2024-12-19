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

    iteration:          int
):

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



    mlflow.log_metric("Reconstruction error", reconstruction_error, step=iteration)
    mlflow.log_metric("Graph regularization", graph_regularization, step=iteration)
    mlflow.log_metric("Sparsity control for Ws", sparsity_control_for_Ws, step=iteration)
    mlflow.log_metric("Sparsity control for H", sparsity_control_for_H, step=iteration)

    return f



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
        Belle = W @ H @ H.T + alpha * cp.sum(cp.array([degree_block[d][p] @ Wp for p, Wp in enumerate(Ws_current)]), axis=0) + betas[d]

        next_W = Ariel / Belle * W
        next_Ws.append(next_W)


    Ariel = cp.sum(cp.array([W.T @ X for W, X in zip(next_Ws, Xs)]), axis=0)
    Cindy = cp.sum(cp.array([W.T @ W @ H for W in next_Ws]), axis=0) + cp.broadcast_to(gammas, (H.shape[0], H.shape[1]))
    next_H = Ariel / Cindy * H
    


    return next_Ws, next_H

