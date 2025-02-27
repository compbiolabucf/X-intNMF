# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: main_survival_pdf_api.py
# Date: 2025/02/26 23:03:55
# Description: APIs for SA result download
# 
# (c) 2025 bu1th4nh. All rights reserved. 
# Written with dedication in the University of Central Florida, EPCOT and the Magic Kingdom.
# -----------------------------------------------------------------------------------------------


import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Union, Literal

from log_config import initialize_logging
import matplotlib.pyplot as plt

import sys
import pymongo
import logging
import uvicorn
import numpy as np
import pandas as pd
from tqdm import tqdm
from io import BytesIO
from fastapi import FastAPI
from s3fs import S3FileSystem
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_utils.inferring_router import InferringRouter
from contextlib import asynccontextmanager
from colorlog import ColoredFormatter



import tempfile

import lifelines
import sksurv
import warnings
from lifelines.statistics import logrank_test

from sklearn import set_config
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sksurv.linear_model import CoxnetSurvivalAnalysis

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
    'key': 'bu1th4nh',
    'secret': 'ariel.anna.elsa',
    'endpoint_url': 'http://localhost:9000',
}
# -----------------------------------------------------------------------------------------------
# App Startup routine
# -----------------------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_logging()
    yield



# -----------------------------------------------------------------------------------------------
# App 
# -----------------------------------------------------------------------------------------------
router = InferringRouter()
app = FastAPI(
    title="NMF Survival PDF API",
    description="API for NMF Survival PDF",
    version="3.2.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)



# -----------------------------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------------------------
@router.post("/survival_pdf")
def get_pdf(input: dict):
    # logging.fatal(str(input))
    omics_mode = input['omics_mode']
    disease = input['dataset_id']
    surv_target = input['surv_target']
    cfg = input['best_cfg']


    best_alpha = input['best_alpha']
    train_sample_ids = input['train_sample_ids']
    test_sample_ids = input['test_sample_ids']



    # Data directories
    if omics_mode == "3-omic":
        surv_target_folder = 'survivalanalysis_testdata_3_omics_mRNA_miRNA_methDNA'
        H_top_level_folders = 'SimilarSampleCrossOmicNMF_3Omics'
    elif omics_mode == "2-omic":
        surv_target_folder = 'survivalanalysis_testdata_2_omics_mRNA_miRNA'
        H_top_level_folders = 'SimilarSampleCrossOmicNMF'


    if disease == "BRCA":
        H_folders = 'brca'
        dataset_folder = 'BreastCancer'
    elif disease == "LUAD":
        H_folders = 'luad'
        dataset_folder = 'LungCancer'
    elif disease == "OV":
        H_folders = 'ov'
        dataset_folder = 'OvarianCancer'


    if surv_target == 'survival': 
        event_label = 'Overall Survival Status'
        duration_label = 'Overall Survival (Months)'
    else:
        event_label = 'Disease Free Status'
        duration_label = 'Disease Free (Months)'

    # Load data
    logging.info("Loading data")
    H = pd.read_parquet(f's3://results/{H_top_level_folders}/{H_folders}/{cfg}/H.parquet', storage_options=storage_options)
    survival_data = pd.read_parquet(f's3://datasets/{dataset_folder}/{surv_target_folder}/{surv_target}.parquet', storage_options=storage_options)


    # Train model
    X_train = H.loc[train_sample_ids, :].values
    Y_train = survival_data.loc[train_sample_ids, [event_label, duration_label]].to_records(index=False)


    X_test  = H.loc[test_sample_ids, :].values
    Y_test  = survival_data.loc[test_sample_ids, [event_label, duration_label]].to_records(index=False)

    logging.info(f"Fitting final model with alpha={best_alpha}")
    coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=[best_alpha]))
    coxnet_pipe.fit(X_train, Y_train)



    # -----------------------------------------------------------------------------------------------
    # Prognostic index and low/hi risk group
    # -----------------------------------------------------------------------------------------------
    logging.info("Getting prognostic index and low/hi risk group")
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
    logging.info("Getting Kaplan-Meier curve")
    high_risk_time_exit = survival_data.loc[high_risk_ids, duration_label].values
    high_risk_event_observed = survival_data.loc[high_risk_ids, event_label].values
    kmf_high = lifelines.KaplanMeierFitter()
    kmf_high.fit(
        high_risk_time_exit, 
        high_risk_event_observed,
        label=f'High Risk ({len(high_risk_ids)})'
    )


    low_risk_time_exit = survival_data.loc[low_risk_ids, duration_label].values
    low_risk_event_observed = survival_data.loc[low_risk_ids, event_label].values
    kmf_low = lifelines.KaplanMeierFitter()
    kmf_low.fit(
        low_risk_time_exit, 
        low_risk_event_observed,
        label=f'Low Risk ({len(high_risk_ids)})'
    )

    results = logrank_test(
        low_risk_time_exit, 
        high_risk_time_exit, 
        event_observed_A=low_risk_event_observed, 
        event_observed_B=high_risk_event_observed
    )

    # -----------------------------------------------------------------------------------------------
    # Plotting and export PDF
    # -----------------------------------------------------------------------------------------------
    logging.info("Plotting and exporting PDF")

    # Create a new figure
    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot the Kaplan-Meier curves
    kmf_high.plot(ax=ax, color='red', linestyle='-',  show_censors = True, ci_show = False, censor_styles={'ms': 5, 'marker': 'x', 'markeredgecolor': '#000000'})
    kmf_low.plot(ax=ax, color='blue', linestyle='--', show_censors = True, ci_show = False, censor_styles={'ms': 5, 'marker': 'x', 'markeredgecolor': '#000000'})

    # Add labels and title
    ax.set_xlabel('Time (Months)')
    ax.set_ylabel('Survival Probability')
    ax.set_title(f'Kaplan-Meier Curves for {disease} - { "Survival" if surv_target == "survival" else "Disease Free" }')

    # Add legend
    ax.legend(loc='best')

    # Add x and y axis limits
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 1)

    # Add p-value to the plot
    p_value = results.p_value
    ax.text(0.03, 0.05, f'p-value: {p_value:.4f}', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    # Save the plot as a PDF object and return it through API

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        fig.savefig(tmpfile.name, format='pdf')
        tmpfile.seek(0)
        return FileResponse(tmpfile.name, media_type='application/pdf', filename='survival_curve.pdf')



# -----------------------------------------------------------------------------------------------
# Requests
# -----------------------------------------------------------------------------------------------
app.get("/")
def index(): return {"message": "Welcome to NMF AI Service!"}

app.include_router(router)



# -----------------------------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    uvicorn.run("main_survival_pdf_api:app", host='0.0.0.0', port=6789, reload=True)
