# %%
# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: playground_survival.ipynb
# Date: 2025/02/05 17:30:55
# Description: 
# 
# (c) 2025 bu1th4nh. All rights reserved. 
# Written with dedication in the University of Central Florida, EPCOT and the Magic Kingdom.
# -----------------------------------------------------------------------------------------------

import warnings
warnings.simplefilter("ignore")

import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Union, Literal
import warnings 

from s3fs import S3FileSystem
import numpy as np
import pandas as pd 

from sklearn import set_config
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import FitFailedWarning

from sksurv.linear_model import CoxnetSurvivalAnalysis
import lifelines




key = 'bu1th4nh'
secret = 'ariel.anna.elsa'
endpoint_url = 'http://localhost:9000'

s3 = S3FileSystem(
    anon=False, 
    endpoint_url=endpoint_url,
    key=key,
    secret=secret,
    use_ssl=False
)
storage_options = {
    'key': key,
    'secret': secret,
    'endpoint_url': endpoint_url,
}

DATA_PATH = 's3://datasets/BreastCancer'
RESU_PATH = 's3://results/SimilarSampleCrossOmicNMF/brca'

# %%
survival = pd.read_parquet(f'{DATA_PATH}/processed_2_omics_mRNA_miRNA/survival.parquet', storage_options=storage_options)
H = pd.read_parquet(f'{RESU_PATH}/k-10-alpha-1000-beta-1-gamma-overridden/H.parquet', storage_options=storage_options)

# %%
survival['Overall Survival Status'] = survival['Overall Survival Status'].replace({'0:LIVING': False, '1:DECEASED': True}).astype(bool)
survival['Disease Free Status'] = survival['Disease Free Status'].replace({'0:DiseaseFree': False, '1:Recurred/Progressed': True}).astype(bool)
survival.dropna(inplace=True)

# display(survival.head())
# display(H.head())

# %%
survival = survival[survival['Disease Free (Months)'] > 0]


print(survival.shape)
print(H.shape)

print(len(common_sample_id := list(set(H.index) & set(survival.index))))




# %%



while True:
    warnings.simplefilter("ignore", UserWarning)
    train_sample_ids, test_sample_ids = train_test_split(common_sample_id, test_size=0.2)
    print(len(train_sample_ids), len(test_sample_ids))

    X_train = H.loc[train_sample_ids, :].values
    X_test = H.loc[test_sample_ids, :].values

    Y_train = (
        survival
        .loc[train_sample_ids, ['Overall Survival Status', 'Overall Survival (Months)']]
        .to_records(index=False)
    )

    # print(Y_train)
    # print(X_train)

    # Get possible alphas
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
    print(gcv.best_estimator_.named_steps['coxnetsurvivalanalysis'].alphas[0])
    alpha = gcv.best_estimator_.named_steps['coxnetsurvivalanalysis'].alphas[0]

    # Fit final model
    coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=[alpha]))
    coxnet_pipe.fit(X_train, Y_train)


    # Get prognostic index
    Y_pred = pd.Series(
        coxnet_pipe.predict(X_test),
        index = test_sample_ids,
    )
    prognostic_index = Y_pred.median()


    # Split test set into high/low risk groups
    high_risk_ids = Y_pred[Y_pred > prognostic_index].index
    low_risk_ids = Y_pred[Y_pred <= prognostic_index].index
    print(len(high_risk_ids), len(low_risk_ids))


    # Kaplan-Meier estimator
    high_risk_time_exit = survival.loc[high_risk_ids, 'Disease Free (Months)'].values
    high_risk_event_observed = survival.loc[high_risk_ids, 'Disease Free Status'].values
    kmf_high = lifelines.KaplanMeierFitter()
    kmf_high.fit(
        high_risk_time_exit, 
        high_risk_event_observed,
        label='High-Risk Group'
    )



    low_risk_time_exit = survival.loc[low_risk_ids, 'Disease Free (Months)'].values
    low_risk_event_observed = survival.loc[low_risk_ids, 'Disease Free Status'].values
    kmf_low = lifelines.KaplanMeierFitter()
    kmf_low.fit(
        low_risk_time_exit, 
        low_risk_event_observed,
        label='Low-Risk Group'
    )
    # kmf_low.plot(show_censors = True)
    # kmf_high.plot(show_censors = True)


    # P-value
    from lifelines.statistics import logrank_test
    results = logrank_test(
        low_risk_time_exit, 
        high_risk_time_exit, 
        event_observed_A=low_risk_event_observed, 
        event_observed_B=high_risk_event_observed
    )
    # print(f'P-value: {results.p_value}')

    # draw p-value into plot
    # import matplotlib.pyplot as plt
    # plt.text(0.6, 0.2, f'P-value: {results.p_value:.4f}', transform=plt.gca().transAxes)
    # plt.show()


    # C-index
    from sksurv.metrics import concordance_index_censored
    c_index = concordance_index_censored(
        survival.loc[test_sample_ids, 'Disease Free Status'].values, 
        survival.loc[test_sample_ids, 'Disease Free (Months)'].values, 
        Y_pred
    )

    # print(f'Concordance index: {c_index[0]}')

    if (results.p_value < 0.01): 
        print('Found!')
        kmf_low.plot(show_censors = True)
        kmf_high.plot(show_censors = True)
        # draw p-value into plot
        import matplotlib.pyplot as plt
        plt.text(0.6, 0.2, f'P-value: {results.p_value:.4f}', transform=plt.gca().transAxes)
        plt.show()
        raise ValueError('Done.')
    

    # break


