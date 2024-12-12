# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: main_mofa2.py
# Date: 2024/12/12 15:40:28
# Description: Baseline implementation for MOFA2 against our model
# 
# 
# 
# (c) 2024 bu1th4nh. All rights reserved. 
# Written with dedication in the University of Central Florida, EPCOT and the Magic Kingdom.
# -----------------------------------------------------------------------------------------------


import mlflow
import pymongo
import logging
import numpy as np
import mofax as mfx
import pandas as pd
from tqdm import tqdm
from s3fs import S3FileSystem
from mofapy2.run.entry_point import entry_point
from typing import List, Dict, Any, Tuple, Union, Literal
from downstream.classification import evaluate_one_target

tqdm.pandas()
mlflow.set_tracking_uri('http://localhost:6969')




key = 'bu1th4nh'
secret = 'ariel.anna.elsa'
endpoint_url = 'http://localhost:19000'

s3 = S3FileSystem(
    anon=False, 
    endpoint_url=endpoint_url,
    key=key,
    secret=secret,
    use_ssl=False
)
storage_option = {
    'key': key,
    'secret': secret,
    'endpoint_url': endpoint_url,
}

mongo = pymongo.MongoClient(
    host='mongodb://localhost',
    port=27017,
    username='bu1th4nh',
    password='ariel.anna.elsa',
)
mongo_db = mongo['SimilarSampleCrossOmicNMF']


configs = [
    ('BreastCancer/processed_crossOmics', 'BreastCancer', 'brca', 'BRCA', 'SimilarSampleCrossOmicNMFv3'),
    ('LungCancer/processed', 'LungCancer', 'luad', 'LUAD', 'SimilarSampleCrossOmicNMFv3_LUAD'),
    ('OvarianCancer/processed', 'OvarianCancer', 'ov', 'OV', 'SimilarSampleCrossOmicNMFv3_OV'),
]
mofa_latent_dims = 15

def find_run(collection, run_id: str, target_id: str): return collection.find_one({'run_id': run_id, 'target_id': target_id})

for ds_name, general_data_name, res_folder, mongo_collection, mlf_experiment_name in configs[1:2]:
    DATA_PATH = f's3://datasets/{ds_name}'
    DATA_PATH = f's3://datasets/{ds_name}'
    TARG_PATH = f's3://datasets/{general_data_name}/clinical_testdata'
    DR_RES_PATH = f's3://results/SimilarSampleCrossOmicNMF/{res_folder}/baseline_MOFA2'
    miRNA = pd.read_parquet(f"{DATA_PATH}/miRNA.parquet", storage_options=storage_option)
    mRNA = pd.read_parquet(f"{DATA_PATH}/mRNA.parquet", storage_options=storage_option)

    mlflow.set_experiment(mlf_experiment_name)
    collection = mongo_db[mongo_collection]

    # miRNA.head()
    print("Dataset: ", ds_name.split('/')[0])
    print("miRNA")
    print(f"Sample size: {miRNA.shape[1]}")
    print(f"Feature size: {miRNA.shape[0]}")
    print("mRNA")
    print(f"Sample size: {mRNA.shape[1]}")
    print(f"Feature size: {mRNA.shape[0]}")

    data_mat = [[miRNA.T.values], [mRNA.T.values]]

    Ariel = entry_point()
    Ariel.set_data_matrix(
        data_mat, 
        likelihoods=['gaussian', 'gaussian'], 
        views_names=['miRNA', 'mRNA'],
        features_names=[miRNA.index, mRNA.index],
        samples_names=[miRNA.columns],
    )

    Ariel.set_model_options(
        factors=mofa_latent_dims
    )

    Ariel.set_train_options(
        convergence_mode = "fast",
    )

    Ariel.build()
    Ariel.run()

    Ariel.save("output.hdf5")
    Belle = mfx.mofa_model("output.hdf5").get_factors(factors=range(mofa_latent_dims), df=True)
    
    s3.mkdirs(DR_RES_PATH, exist_ok=True)
    Belle.to_parquet(f"{DR_RES_PATH}/H.parquet", storage_options=storage_option)
    
    run_id = s3.open(f"{DR_RES_PATH}/run_id.txt", 'r').read() if s3.exists(f"{DR_RES_PATH}/run_id.txt") else None

    
    with mlflow.start_run(run_id=run_id) if run_id is not None else mlflow.start_run(run_name='baseline_MOFA2'):

        if run_id is None: 
            mlflow.log_param("Number of omics layers", 2)
            mlflow.log_param("Omics layers feature size", [mRNA.shape[0], miRNA.shape[0]])
            mlflow.log_param("Sample size", miRNA.shape[1])
            mlflow.log_param("Latent size", mofa_latent_dims)


            run_id = mlflow.active_run().info.run_id
            with s3.open(f"{DR_RES_PATH}/run_id.txt", 'w') as f:
                f.write(run_id)
        H = Belle.copy(deep=True)
        target_folders = [f's3://{a}' for a in s3.ls(TARG_PATH)]

        for target_folder in target_folders:
            # Retrieve test data
            target_id = str(target_folder.split('/')[-1]).split('.')[0]
            # if find_run(collection, run_id, target_id) is not None:
            #     logging.info(f"Run {run_id} on dataset {target_id} already exists. Skipping")
            #     continue
            test_data = pd.read_parquet(target_folder, storage_options=storage_option)

            # Evaluate
            result_pack = evaluate_one_target(H, testdata = test_data, methods_list = ["Logistic Regression", "Random Forest"], target = target_id)

            # Load to staging package
            data_pack = {
                'run_id': run_id,
                'target_id': target_id,
                'summary': {}
            }
            for method in result_pack.keys():
                data_pack[method] = result_pack[method].to_dict(orient='index')

            for metric in result_pack[method].columns:
                if str(metric).isupper():
                    # Assume all metrics are upper case-noted columns
                    data_pack['summary'][f'{method} Mean {metric}'] = float(np.mean(result_pack[method][metric].values))
                    data_pack['summary'][f'{method} Median {metric}'] = float(np.median(result_pack[method][metric].values))
                    data_pack['summary'][f'{method} Std {metric}'] = float(np.std(result_pack[method][metric].values))
                    data_pack['summary'][f'{method} Max {metric}'] = float(np.max(result_pack[method][metric].values))
                    data_pack['summary'][f'{method} Min {metric}'] = float(np.min(result_pack[method][metric].values))

            # # Log to MLFlow
            for key in data_pack['summary'].keys():
                if 'Mean AUROC' in key: mlflow.log_metric(f'{target_id} {" ".join(key.split(" ")[:2])} Mean AUC', data_pack['summary'][key])
                if 'Mean MCC' in key: mlflow.log_metric(f'{target_id} {key}', data_pack['summary'][key])
        
        
            # Save to MongoDB
            collection.update_one(
                {'run_id': run_id, 'target_id': target_id},
                {'$set': data_pack},
                upsert=True
            )