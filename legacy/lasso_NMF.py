# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành (@bu1th4nh)
# Title: lasso_NMF.py
# Date: 2024/08/31 15:09:58
# Description: Class implementation for Lasso method for solving NMF problem with 
# 
# (c) bu1th4nh. All rights reserved
# -----------------------------------------------------------------------------------------------

from typing import List, Tuple, Union, Literal, Any, Callable
from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd


class LassoNMF:
    """
        Solving NMF problem for multi-omics data integration using Lasso method     

        Input
        -----


        - Data:
            - X: list of D matrices of shape (m_d, N). m_d can be different for each d-th omic type, but N is the same for all omics. All data MUST be CENTERED

        - Integers:
            - D: number of omic types (will be self-inferrable from the data)
            - N: number of samples (will be self-inferrable from the data)
            - m_d: number of features of the d-th omic type (will be self-inferrable from the data)

        - Hyperparameters:
            - k: number of latent features
            - alpha: L1 regularization parameter for H
            - beta_d: L1 regularization parameter for W_d
        These hyperparameters can be auto-tuned or manually set by the user

        
        Output
        ------
        - W: list of D matrices of shape (m_d, k)
        - H: matrix of shape (k, N)
    """

    omic_data_list: List[np.ndarray]

    num_omics: int
    num_samples: int
    num_features: List[int]

    # -----------------------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------------------
    def __init__(
        self,
        X: List[np.ndarray],
    ) -> None:
        """
            Initialize the LassoNMF object with the input data

            Parameters
            ----------
            X: list of D matrices of shape (m_d, N). m_d can be different for each d-th omic type, but N is the same for all omics        
        """
        self.omic_data_list = X

        self.num_omics = len(X)
        self.num_samples = X[0].shape[1]
        self.num_features = [X[i].shape[0] for i in range(self.num_omics)]


    # -----------------------------------------------------------------------------------------------
    # Solver
    # -----------------------------------------------------------------------------------------------
    def solve_by_fixing_weight(
        self,
        weight_initialization: Literal['random', 'svd', 'clustering', 'custom'] = 'random',
        custom_initialization: Callable[[List[np.ndarray], int, int], Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Any:
        """
        
        
        """
        pass

    
    def solve_by_fixing_basis():
        pass


    # -----------------------------------------------------------------------------------------------
    # Individual Step
    # -----------------------------------------------------------------------------------------------
    def solve_H_with_fixed_Ws(
        self,
        Ws: List[np.ndarray],
        alphas: List[float],
    ) -> np.ndarray:
        """
            Solve the basis matrix H with fixed coefficient Ws

            Parameters
            ----------
            Ws: list of D matrices of shape (m_d, k)
            alphas: list of N L1 regularization parameters for H, with respect to each sample

            Returns
            ----------
            A matrix H with shape (k, N)


        """

        # Concat all omics data matrix along the column axis, aka big_X = [X_1; X_2; ...; X_D] with shape (m_1 + m_2 + ... + m_D, N)
        big_X = np.concatenate(self.omic_data_list, axis=0)
        
        # Concat all Ws data matrix along the column axis, aka big_W = [W_1; W_2; ...; W_D] with shape (m_1 + m_2 + ... + m_D, k)
        big_W = np.concatenate(Ws, axis=0)

        # Initialize H
        H_coeffs = []

        # Solve individual column of H
        for index, alpha in enumerate(alphas):
            lasso = Lasso(alpha = alpha, fit_intercept = False);
            lasso.fit(
                big_W,
                big_X[:, index]
            )
            H_coeffs.append(np.array([lasso.coef_.T]))

        # Construct H
        return np.concatenate(H_coeffs, axis=1)

    def solve_Ws_with_fixed_H(
        self, 
        H: np.ndarray,
        betas: List[float],
    ) -> List[np.ndarray]:
        """
            Solve W individually with H

            Parameters
            ----------
            A matrix H with shape (k, N)
            betas: list of N L1 regularization parameters for each W, with respect to each sample

            Returns
            ----------
            Ws: list of D matrices of shape (m_d, k)

        """

        Ws = []
        for index, omic in self.omic_data_list:
            identity = np.eye(omic.shape[0])
            feature_size = omic.shape[0]

            X_tilded = omic.reshape(-1, 1)
            H_tilded = np.kron(identity, H.T)

            print(X_tilded.shape)
            print(H_tilded.shape)

            lasso = Lasso(betas[index], fit_intercept=False)
            lasso.fit(H_tilded, X_tilded)
            W = np.array(lasso.coef_).reshape((feature_size, -1))

            Ws.append(W)