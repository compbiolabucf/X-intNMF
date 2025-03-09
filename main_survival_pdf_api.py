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
import base64
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

from downstream.survival import surv_analysis

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


    # Get ablated data
    half_before_alpha = cfg.split('-alpha-')[0]
    graph_reg_alpha = cfg.split('-alpha-')[1].split('-', maxsplit = 1)[0]
    half_after_alpha = cfg.split('-alpha-')[1].split('-', maxsplit = 1)[1]
    min_alpha_cfg_id = f"{half_before_alpha}-alpha-0-{half_after_alpha}"



    # Data directories
    if omics_mode == "3-omic":
        surv_target_folder = 'survivalanalysis_testdata_3_omics_mRNA_miRNA_methDNA'
        original_dataset_folder = 'processed_3_omics_mRNA_miRNA_methDNA'
        H_top_level_folders = 'SimilarSampleCrossOmicNMF_3Omics'
    elif omics_mode == "2-omic":
        surv_target_folder = 'survivalanalysis_testdata_2_omics_mRNA_miRNA'
        original_dataset_folder = 'processed_2_omics_mRNA_miRNA'
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
        y_axis_label = 'Survival Probability'
    else:
        event_label = 'Disease Free Status'
        duration_label = 'Disease Free (Months)'
        y_axis_label = 'Disease Free Probability'

    # Load data
    logging.info("Loading data")
    H = pd.read_parquet(f's3://results/{H_top_level_folders}/{H_folders}/{cfg}/H.parquet', storage_options=storage_options)
    H_ablated = pd.read_parquet(f's3://results/{H_top_level_folders}/{H_folders}/{min_alpha_cfg_id}/H.parquet', storage_options=storage_options)
    survival_data = pd.read_parquet(f's3://datasets/{dataset_folder}/{surv_target_folder}/{surv_target}.parquet', storage_options=storage_options)
    aio_data = pd.read_parquet(f's3://datasets/{dataset_folder}/{original_dataset_folder}/allinone.parquet', storage_options=storage_options).T




    # -----------------------------------------------------------------------------------------------
    # Survival analysis 
    # -----------------------------------------------------------------------------------------------
    logging.info("Running survival analysis")
    surv_result = surv_analysis(
        H, 
        train_sample_ids,
        survival_data.loc[train_sample_ids],
        test_sample_ids,
        survival_data.loc[test_sample_ids],
        event_label,
        duration_label,   
        return_kmf_object=True,
        alpha=best_alpha,
    )



    # -----------------------------------------------------------------------------------------------
    # Ablated survival analysis 
    # -----------------------------------------------------------------------------------------------
    logging.info("Running ablated survival analysis")
    surv_ablated = surv_analysis(
        H_ablated, 
        train_sample_ids,
        survival_data.loc[train_sample_ids],
        test_sample_ids,
        survival_data.loc[test_sample_ids],
        event_label,
        duration_label,   
        return_kmf_object=True,
        alpha=best_alpha,
    )


    # -----------------------------------------------------------------------------------------------
    # Baseline survival analysis 
    # -----------------------------------------------------------------------------------------------
    logging.info("Running baseline survival analysis")
    surv_baseline = surv_analysis(
        aio_data, 
        train_sample_ids,
        survival_data.loc[train_sample_ids],
        test_sample_ids,
        survival_data.loc[test_sample_ids],
        event_label,
        duration_label,   
        return_kmf_object=True,
        alpha=0.05,
    )


    # -----------------------------------------------------------------------------------------------
    # Plotting and export PDF
    # -----------------------------------------------------------------------------------------------
    logging.info("Getting Kaplan-Meier curve")

    # Create a new figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the Kaplan-Meier embedding survival curves
    surv_result['kmf_low'].plot(ax=ax[0], color='blue', linestyle='--', show_censors = True, ci_show = False, censor_styles={'ms': 6, 'marker': 'x', 'markeredgecolor': '#000000'})
    surv_result['kmf_high'].plot(ax=ax[0], color='red', linestyle='-',  show_censors = True, ci_show = False, censor_styles={'ms': 6, 'marker': 'x', 'markeredgecolor': '#000000'})

    # Plot the Kaplan-Meier ablated survival curves
    surv_ablated['kmf_low'].plot(ax=ax[1], color='blue', linestyle='--', show_censors = True, ci_show = False, censor_styles={'ms': 6, 'marker': 'x', 'markeredgecolor': '#000000'})
    surv_ablated['kmf_high'].plot(ax=ax[1], color='red', linestyle='-',  show_censors = True, ci_show = False, censor_styles={'ms': 6, 'marker': 'x', 'markeredgecolor': '#000000'})

    # Plot the Kaplan-Meier baseline survival curve
    surv_baseline['kmf_low'].plot(ax=ax[2], color='blue', linestyle='--', show_censors = True, ci_show = False, censor_styles={'ms': 6, 'marker': 'x', 'markeredgecolor': '#000000'})
    surv_baseline['kmf_high'].plot(ax=ax[2], color='red', linestyle='-',  show_censors = True, ci_show = False, censor_styles={'ms': 6, 'marker': 'x', 'markeredgecolor': '#000000'})

    


    # Add titles
    # fig.suptitle(f'Kaplan-Meier Curves for {omics_mode}s - {disease} - { "Survival" if surv_target == "survival" else "Disease Free" }', size = 22)
    fig.subplots_adjust(top=0.85)
    ax[0].set_title(f'X-intNMF', size=22)
    ax[1].set_title(f'X-intNMF (alpha = 0)', size=22)
    ax[2].set_title(f'Original data', size=22)



    # Add x and y axis labels
    ax[0].set_xlabel('Time (Months)', size=22)
    ax[1].set_xlabel('Time (Months)', size=22)
    ax[2].set_xlabel('Time (Months)', size=22)
    ax[0].set_ylabel(y_axis_label, size=22)


    # Add legend
    ax[0].legend(loc='best')
    ax[1].legend(loc='best')
    ax[2].legend(loc='best')

    # Add x and y axis limits
    ax[0].set_xlim(0, 120)
    ax[0].set_ylim(0, 1.02)
    ax[1].set_xlim(0, 120)
    ax[1].set_ylim(0, 1.02)
    ax[2].set_xlim(0, 120)
    ax[2].set_ylim(0, 1.02)

    # Add a)/b)/c) labels outside the plot area
    fig.text(0.103, 0.91, '(a)', fontsize=16, verticalalignment='top', weight='bold')
    fig.text(0.377, 0.91, '(b)', fontsize=16, verticalalignment='top', weight='bold')
    fig.text(0.652, 0.91, '(c)', fontsize=16, verticalalignment='top', weight='bold')


    # Add p-value to the plot
    p_value = surv_result['logrank_result'].p_value
    ax[0].text(0.03, 0.05, f'p-value: {p_value:.5f}', transform=ax[0].transAxes, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'), size=16)
    p_value = surv_ablated['logrank_result'].p_value
    ax[1].text(0.03, 0.05, f'p-value: {p_value:.5f}', transform=ax[1].transAxes, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'), size=16)
    p_value = surv_baseline['logrank_result'].p_value
    ax[2].text(0.03, 0.05, f'p-value: {p_value:.5f}', transform=ax[2].transAxes, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'), size=16)


    # Save the plot as a PDF object and return it through API
    pdf_bytes_base64 = None
    logging.info("Plotting and exporting PDF")
    with tempfile.NamedTemporaryFile(prefix='CrossOmicResult_', suffix=".pdf") as tmpfile:
        fig.savefig(tmpfile.name, format='pdf', bbox_inches='tight')
        tmpfile.seek(0)
        pdf_bytes_base64 = base64.b64encode(tmpfile.read()).decode('utf-8')

    return {
        'kmf_plot': pdf_bytes_base64,
        'baseline_pvalue': surv_baseline['logrank_result'].p_value,
        'ablated_pvalue': surv_ablated['logrank_result'].p_value,
    }


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
