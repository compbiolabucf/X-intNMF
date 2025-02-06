# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: survival.py
# Date: 2025/02/02 12:15:54
# Description: 
# 
# (c) 2025 bu1th4nh. All rights reserved. 
# Written with dedication in the University of Central Florida, EPCOT and the Magic Kingdom.
# -----------------------------------------------------------------------------------------------


import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Union, Literal

import lifelines
import sksurv
import warnings

from sklearn import set_config
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sksurv.linear_model import CoxnetSurvivalAnalysis



def surv_analysis(H, train_sample_ids, train_surv_data, test_sample_ids, test_surv_data):
    """
        Perform survival analysis using Cox Proportional Hazard with Elastic Net regularization.

        Parameters
        ----------

        `H`: `pd.DataFrame`
            Sample latent representation
        `train_sample_ids`: `List[str]`
            Sample IDs for training
        `train_surv_data`: `List[Tuple[bool, float]]`
            Survival data corresponding to `train_sample_ids`. Each element is a tuple of (event, time), where event is a boolean indicating whether the event occurred, and time is the time until the event or censoring.
        `test_sample_ids`: `List[str]`
            Sample IDs for testing
        `test_surv_data`: `List[Tuple[bool, float]]`
            Survival data corresponding to `test_sample_ids`. The same as `train_surv_data`.

        Returns
        -------
        - C-index on test
        - Brier score on test
        - p-value for log-rank test on hi and lo PI index group
        - Prognostic index
        - Coefficients (coeff, alpha, r)
        - Kaplan-Meier curve for hi and lo PI index group
    """

    # Fit model

    X_train = H.loc[train_sample_ids, :].values
    coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01, max_iter=100))
    coxnet_pipe.fit(X_train, train_surv_data)

    estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    gcv = GridSearchCV(
        make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9)),
        param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
        cv=cv,
        error_score=0.5,
        n_jobs=3,
    ).fit(X_train, train_surv_data)

    best_alpha = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"].alphas_[0]
    best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
    coeff = best_model.coef_
    r = best_model.r_
    alpha = best_alpha
    logging.info(f"Best alpha: {best_alpha}")
    logging.info(f"Coefficients: {coeff}")
    logging.info(f"r: {r}")
    logging.info(f"r.shape: {r.shape}")

    # Get prognostic index
    X_test = H.loc[test_sample_ids, :].values
    Y_test = best_model.predict(X_test)
    prog_index = np.median(Y_test)

    # Divide into hi and lo groups
    hi_group = test_sample_ids[Y_test > prog_index]
    lo_group = test_sample_ids[Y_test <= prog_index]

    # Kaplan-Meier curve
    kmf_hi = lifelines.KaplanMeierFitter()
    kmf_lo = lifelines.KaplanMeierFitter()
    kmf_hi.fit(test_surv_data[Y_test > prog_index], event_observed=test_surv_data[Y_test > prog_index][:, 0])
    kmf_lo.fit(test_surv_data[Y_test <= prog_index], event_observed=test_surv_data[Y_test <= prog_index][:, 0])
    






