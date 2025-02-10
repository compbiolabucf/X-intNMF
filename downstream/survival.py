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



def surv_analysis(H, train_sample_ids, train_surv_data, test_sample_ids, test_surv_data, event_label, duration_label):
    """
        Perform survival analysis using Cox Proportional Hazard with Elastic Net regularization.

        Parameters
        ----------

        `H`: `pd.DataFrame`
            Sample latent representation
        `train_sample_ids`: `List[str]`
            Sample IDs for training
        `train_surv_data`: `pd.DataFrame`
            Survival data corresponding to `train_sample_ids`. Each element is a tuple of (event, time), where event is a boolean indicating whether the event occurred, and time is the time until the event or censoring.
        `test_sample_ids`: `List[str]`
            Sample IDs for testing
        `test_surv_data`: `pd.DataFrame`
            Survival data corresponding to `test_sample_ids`. The same as `train_surv_data`.
        `event_label`: `str`
            Column name for event status, where True indicates the event occurred and False indicates censoring.
        `duration_label`: `str`
            Column name for duration until the event or censoring.

        Returns
        -------
        - C-index on test
        - Integrated Brier score on test
        - p-value for log-rank test on hi and lo PI index group
        - Prognostic index
        - Coefficients (coeff, alpha, r)
        - Kaplan-Meier curve for hi and lo PI index group
    """

    
    X_train = H.loc[train_sample_ids, :].values
    X_test  = H.loc[test_sample_ids, :].values
    Y_train = train_surv_data.to_records(index=False)


    # print(Y_train)
    # print(X_train)

    # -----------------------------------------------------------------------------------------------
    # Coxnet with CV
    # -----------------------------------------------------------------------------------------------
    coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01, max_iter=100))
    coxnet_pipe.fit(X_train, Y_train)
    estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_

    # 5-fold CV to find optimal alpha
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    gcv = GridSearchCV(
        make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.5)),
        param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
        cv=cv,
        error_score=0.5,
        n_jobs=3,
    ).fit(X_train, Y_train)
    # print(gcv.best_estimator_.named_steps['coxnetsurvivalanalysis'].alphas[0])
    alpha = gcv.best_estimator_.named_steps['coxnetsurvivalanalysis'].alphas[0]

    # Fit final model
    coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=[alpha]))
    coxnet_pipe.fit(X_train, Y_train)


    # -----------------------------------------------------------------------------------------------
    # Prognostic index and low/hi risk group
    # -----------------------------------------------------------------------------------------------
    # Get prognostic index
    Y_pred = pd.Series(
        coxnet_pipe.predict(X_test),
        index = test_sample_ids,
    )
    prognostic_index = Y_pred.median()


    # Split test set into high/low risk groups
    high_risk_ids = Y_pred[Y_pred > prognostic_index].index
    low_risk_ids = Y_pred[Y_pred <= prognostic_index].index
    # print(len(high_risk_ids), len(low_risk_ids))


    # -----------------------------------------------------------------------------------------------
    # Kaplan-Meier curve
    # -----------------------------------------------------------------------------------------------
    high_risk_time_exit = test_surv_data.loc[high_risk_ids, duration_label].values
    high_risk_event_observed = test_surv_data.loc[high_risk_ids, event_label].values
    kmf_high = lifelines.KaplanMeierFitter()
    kmf_high.fit(
        high_risk_time_exit, 
        high_risk_event_observed,
        label='High-Risk Group'
    )


    low_risk_time_exit = test_surv_data.loc[low_risk_ids, duration_label].values
    low_risk_event_observed = test_surv_data.loc[low_risk_ids, event_label].values
    kmf_low = lifelines.KaplanMeierFitter()
    kmf_low.fit(
        low_risk_time_exit, 
        low_risk_event_observed,
        label='Low-Risk Group'
    )
    


    # Get X-Y, censor point data from kmf_high and kmf_low
    kmf_high_survival_function = kmf_high.survival_function_
    kmf_low_survival_function = kmf_low.survival_function_

    # Extract X-Y data
    X_high = kmf_high_survival_function.index.values
    Y_high = kmf_high_survival_function['High-Risk Group'].values

    X_low = kmf_low_survival_function.index.values
    Y_low = kmf_low_survival_function['Low-Risk Group'].values

    # Extract censor points
    censor_high = kmf_high.event_table[kmf_high.event_table['censored'] > 0].index.values
    censor_low = kmf_low.event_table[kmf_low.event_table['censored'] > 0].index.values
    censor_high_pred = kmf_high.predict(censor_high)
    censor_low_pred = kmf_low.predict(censor_low)




    # -----------------------------------------------------------------------------------------------
    # P-value
    # -----------------------------------------------------------------------------------------------
    from lifelines.statistics import logrank_test
    results = logrank_test(
        low_risk_time_exit, 
        high_risk_time_exit, 
        event_observed_A=low_risk_event_observed, 
        event_observed_B=high_risk_event_observed
    )
    





    # -----------------------------------------------------------------------------------------------
    # Integrated Brier score
    # -----------------------------------------------------------------------------------------------
    Brier_score = np.nan





    # -----------------------------------------------------------------------------------------------
    # C-index
    # -----------------------------------------------------------------------------------------------
    c_index = np.nan




    # -----------------------------------------------------------------------------------------------
    # Return
    # -----------------------------------------------------------------------------------------------
    return {
        "train_sample_ids": list(train_sample_ids),
        "test_sample_ids": list(test_sample_ids),
        "test_pred": list(Y_pred),
        "test_prognostic_index": prognostic_index,
        "test_low_risk_ids": list(low_risk_ids),
        "test_high_risk_ids": list(high_risk_ids),
        "kaplan_meier_curve": {
            "X_high": list(X_high),
            "Y_high": list(Y_high),
            "censor_high": list(censor_high),
            "censor_high_pred": list(censor_high_pred),
            "X_low": list(X_low),
            "Y_low": list(Y_low),
            "censor_low": list(censor_low),
            "censor_low_pred": list(censor_low_pred),
        },
        "p_value": results.p_value,
        "c_index": c_index,
        "integrated_brier_score": Brier_score,
    }