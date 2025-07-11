# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành (@bu1th4nh)
# Title: _supports.py
# Date: 2024/09/17 14:41:36
# Description: Contains the supporting functions for the main functions in the package
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


import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Literal, Any, Callable, Dict

# @jit
def compute_similarity(
    self,
    omics_layers: np.ndarray, 
    method: Union[Callable, None] = None,
    get_abs_value: bool = True
) -> np.ndarray:
    """
        Calculate the similarity/interaction matrix between features in the omics layer X (1). The similarity score can further take the absolute value if `get_abs_value` is True
        
        Input
        -----
        `omics_layers`: np.ndarray
            The omics layers matrix of shape (feature, sample), or (m, N)
        `method`: Union[Callable, None]
            The similarity metric to calculate the similarity matrix. If None, the default metric is Pearson correlation coefficient. The method should take the omics layers matrix as input and return the similarity matrix of shape (feature, feature), or (m, m)
        `get_abs_value`: bool
            Whether to take the absolute value of the similarity matrix or not


        Output
        ------
        `similarity_matrix`: np.ndarray
            The similarity matrix of shape (feature, feature), or (m, m)
    """
    
    corr = np.corrcoef(omics_layers) if method is None else method(omics_layers)

    logging.info(f"Corr matrix shape: {corr.shape}")

    return np.abs(corr) if get_abs_value else corr



# @jit
def divisor_func_selection(
    self,
    method: Literal["l2", "l1", "max", "min", "mean", "median", "sum"],
    axis: Literal["row", "column", "all"]
) -> Callable:
    
    if axis == "row": orientation, shape = 1, (-1, 1)
    elif axis == "column": orientation, shape = 0, (1, -1)
    else: orientation, shape = None, -1

    # logging.info(orientation, shape)

    if method == "l2":       return lambda x: np.reshape(np.linalg.norm(x, ord=2, axis=orientation), newshape = shape)
    elif method == "l1":     return lambda x: np.reshape(np.linalg.norm(x, ord=1, axis=orientation), newshape = shape)
    elif method == "max":    return lambda x: np.reshape(np.max(x, axis=orientation), newshape = shape)
    elif method == "min":    return lambda x: np.reshape(np.min(x, axis=orientation), newshape = shape)
    elif method == "mean":   return lambda x: np.reshape(np.mean(x, axis=orientation), newshape = shape)
    elif method == "median": return lambda x: np.reshape(np.median(x, axis=orientation), newshape = shape)
    elif method == "sum":    return lambda x: np.reshape(np.sum(x, axis=orientation), newshape = shape)
    else: return None



# @jit
def norm_func_selection(
    self,
    method: Union[Callable, Literal["l2", "l1", "max", "min", "mean", "median", "sum", "passthru"]],
    orientation: Literal["row", "column", "all"]
) -> Callable:
    if method == "passthru": return lambda x: x
    elif callable(method): return method
    else: 
        div = self.divisor_func_selection(method, orientation)
        return lambda x: x / div(x)



# @jit
def threshold_cutoff(
    self,
    matrix: np.ndarray,
    threshold: float,
):
    """
        Apply threshold cutoff to the matrix
    """
    return matrix * (matrix > threshold)



# @jit
def density_func_selection(
    self,
    density_measure: Union[Callable, Literal['density', 'gini', 'rank']]
):
    """
        Select the density measure function based on the input string.
        If the input is a callable function, return the function itself
        The default density measure is the density function
    """
    if density_measure == "gini": return lambda x: 1 - np.nansum(x ** 2) / np.nansum(x) ** 2
    elif density_measure == "rank": return lambda x: np.nansum(x > 0) / x.size
    elif callable(density_measure): return density_measure
    else: return lambda x: np.count_nonzero(x) / x.size


# @jit
def density_agg_func_selection(
    self,
    agg_method: Literal['mean', 'median', 'max', 'min']
):
    """
        Select the aggregation function based on the input string
    """
    if agg_method == "mean": return np.nanmean
    elif agg_method == "median": return np.nanmedian
    elif agg_method == "max": return np.nanmax
    elif agg_method == "min": return np.nanmin
    else: return None


# @jit
def convert_block_matrix_to_table_of_matrices(
    self,
    block_matrix: np.ndarray,
    size_list: List[int],
    flavor: Literal["numpy", "cupy", "pytorch"] = "numpy"
) -> List[List[np.ndarray]]:
    """
        Convert the 2-D block matrix to the table of matrices

        Input
        -----
        `block_matrix`: np.ndarray
            The block matrix of shape (M, M)
        `size_list`: List[int]
            The list of sizes of each block matrix in the table of matrices with length D
        
        Output
        ------
        `table_of_matrices`: List[List[np.ndarray]]
            The table of matrices of shape (D, D, m_i, m_j)
    """

    if flavor == "numpy": 
        return [
            list(np.hsplit(horizontal_block, size_list))
            for horizontal_block 
            in np.vsplit(block_matrix, size_list)
        ]
    elif flavor == "cupy":
        import cupy as cp
        return [
            list(cp.hsplit(horizontal_block, size_list))
            for horizontal_block 
            in cp.vsplit(block_matrix, size_list)
        ]
    elif flavor == "pytorch":
        import torch
        return [
            list(torch.hsplit(horizontal_block, size_list))
            for horizontal_block 
            in torch.vsplit(block_matrix, size_list)
        ]



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
                                                                                                                                       
