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
import numpy as np
import pandas as pd
import logging



def objective_function(
    self,

    Xs:                 List[np.ndarray], 
    Ws:                 List[np.ndarray], 
    H:                  np.ndarray, 
    similarity_block:   List[List[np.ndarray]], 
    degree_block:       List[List[np.ndarray]], 
    
    alpha:              float, 
    betas:              Union[np.ndarray, List[float]],
    gammas:             Union[np.ndarray, List[float]]
):
    X_concatted = np.concatenate(Xs, axis=0)
    W_concatted = np.concatenate(Ws, axis=0)


    # print(f"Xs: {Xs}")
    # print(f"Ws: {Ws}")
    # print(f"H: {H}")
    # print(f"Sim: {np.block(similarity_block)}")
    # print(f"Deg: {np.block(degree_block)}")

    
    # Nullity check
    self.nullityCheck(X_concatted, additional_info="X_concatted")
    self.nullityCheck(W_concatted, additional_info="W_concatted")
    self.nullityCheck(H, additional_info="H")
    self.nullityCheck(np.block(similarity_block), additional_info="similarity_block")
    self.nullityCheck(np.block(degree_block), additional_info="degree_block")

    # Non-negativity check
    self.negativeCheck(X_concatted, additional_info="X_concatted")
    self.negativeCheck(W_concatted, additional_info="W_concatted")
    self.negativeCheck(H, additional_info="H")
    self.negativeCheck(np.block(similarity_block), additional_info="similarity_block")
    self.negativeCheck(np.block(degree_block), additional_info="degree_block")


    # Calculate the reconstruction error
    reconstruction_error = np.linalg.norm(X_concatted - W_concatted @ H, ord="fro") ** 2

    # Graph regularization term
    laplacian_block = np.block(degree_block) - np.block(similarity_block)
    graph_regularization = np.trace(W_concatted.T @ laplacian_block @ W_concatted)

    # Sparsity control term for W
    sparsity_control_for_Ws = np.array([np.linalg.norm(W, ord=1) for W in Ws]) @ np.array(betas)

    # Sparsity control term for H
    sparsity_control_for_H = np.linalg.norm(H, ord=1, axis = 0) @ np.array(gammas)
    f = 1/2 * reconstruction_error + alpha/2 * graph_regularization + sparsity_control_for_Ws + sparsity_control_for_H

    # logging.info(f"Reconstruction error: {reconstruction_error}")
    # logging.info(f"Graph regularization: {graph_regularization}")
    # logging.info(f"Sparsity control for Ws: {sparsity_control_for_Ws}")
    # logging.info(f"Sparsity control for H: {sparsity_control_for_H}")
    # logging.info(f"Objective function value: {f}")
    return f




def update_Ws(
    self,

    Xs:                 List[np.ndarray], 
    Ws_current:         List[np.ndarray], 
    H:                  np.ndarray, 
    similarity_block:   List[List[np.ndarray]], 
    degree_block:       List[List[np.ndarray]], 

    alpha:              float, 
    betas:              Union[np.ndarray, List[float]],
):
    next_Ws = []
    for d, W in enumerate(Ws_current):
        Ariel = Xs[d] @ H.T + alpha * np.sum([similarity_block[d][p] @ Wp for p, Wp in enumerate(Ws_current)], axis=0)
        Belle = W @ H @ H.T + alpha * np.sum([degree_block[d][p] @ Wp for p, Wp in enumerate(Ws_current)], axis=0) + betas[d]

        next_W = Ariel / Belle * W
        next_Ws.append(next_W)

    # # Nullity check
    # for d, W in enumerate(next_Ws):
    #     self.nullityCheck(W, additional_info=f"W[{d}]")

    # # Non-negativity check
    # for d, W in enumerate(next_Ws):
    #     self.negativeCheck(W, additional_info=f"W[{d}]")

    return next_Ws

