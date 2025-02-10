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



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, precision_score
from sklearn.svm import SVC

tqdm.pandas()
mlflow.set_tracking_uri('http://localhost:6969')


def evaluate_one_target(H, testdata, methods_list, target):
    # Prepping the data and result
    logging.info("Starting evaluation")


    results = {
        method: {} for method in methods_list
    }

    # Iterate through each test
    for test_id in tqdm(testdata.index, desc=f"Evaluating target {target} on testdata"):
        # Get sample IDs
        train_sample_ids = testdata.loc[test_id, f'train_sample_ids']
        train_gnd_truth = testdata.loc[test_id, f'train_ground_truth']
        test_sample_ids = testdata.loc[test_id, f'test_sample_ids']
        test_gnd_truth = testdata.loc[test_id, f'test_ground_truth']

        # Get train test X/Y
        X_train = H.loc[train_sample_ids].values
        Y_train = np.array(train_gnd_truth)
        X_test = H.loc[test_sample_ids].values
        Y_test = np.array(test_gnd_truth)

        # Evaluate each method
        for cls_method in methods_list:
            if(cls_method == "SVM"):                    cls = SVC(probability=True, verbose=False)
            elif(cls_method == "Random Forest"):        cls = RandomForestClassifier(verbose=False)
            elif(cls_method == "Logistic Regression"):  cls = LogisticRegression(max_iter=1000, verbose=False)
            elif(cls_method == "AdaBoost"):             cls = AdaBoostClassifier()

            # Fit & predict the model
            cls.fit(X_train, Y_train)
            pred = cls.predict(X_test)
            prob = cls.predict_proba(X_test)[::,1]

            # Metrics
            ACC = accuracy_score(Y_test, pred)
            PRE = precision_score(Y_test, pred)
            REC = recall_score(Y_test, pred)
            F1 = f1_score(Y_test, pred)
            MCC = matthews_corrcoef(Y_test, pred)
            AUROC = roc_auc_score(Y_test, prob)
            AUPRC = average_precision_score(Y_test, prob)

            # Store the result
            results[cls_method][test_id] = {
                'pred': pd.Series(pred).astype(int).tolist(),
                'prob': pd.Series(prob).astype(float).tolist(),
                'ACC': float(ACC),
                'PRE': float(PRE),
                'REC': float(REC),
                'F1': float(F1),
                'MCC': float(MCC),
                'AUROC': float(AUROC),
                'AUPRC': float(AUPRC),
            }

    return results

            
      

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
mongo_db = mongo['SimilarSampleCrossOmicNMF_3Omics']


configs = [
    ('BreastCancer/processed_3_omics_mRNA_miRNA_methDNA', 'BreastCancer', 'brca', 'BRCA', 'SimilarSampleCrossOmicNMFv3_BRCA_3Omics'),
    ('LungCancer/processed_3_omics_mRNA_miRNA_methDNA', 'LungCancer', 'luad', 'LUAD', 'SimilarSampleCrossOmicNMFv3_LUAD_3Omics'),
    ('OvarianCancer/processed_3_omics_mRNA_miRNA_methDNA', 'OvarianCancer', 'ov', 'OV', 'SimilarSampleCrossOmicNMFv3_OV_3Omics'),
]
mofa_latent_dims = 15

def find_run(collection, run_id: str, target_id: str): return collection.find_one({'run_id': run_id, 'target_id': target_id})

for ds_name, general_data_name, res_folder, mongo_collection, mlf_experiment_name in configs:
    DATA_PATH = f's3://datasets/{ds_name}'
    TARG_PATH = f's3://datasets/{general_data_name}/clinical_testdata_3_omics_mRNA_miRNA_methDNA'
    DR_RES_PATH = f's3://results/SimilarSampleCrossOmicNMF_3Omics/{res_folder}/baseline_MOFA2'
    miRNA = pd.read_parquet(f"{DATA_PATH}/miRNA.parquet", storage_options=storage_option)
    mRNA = pd.read_parquet(f"{DATA_PATH}/mRNA.parquet", storage_options=storage_option)
    methDNA = pd.read_parquet(f'{DATA_PATH}/methDNA.parquet', storage_options=storage_option)

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
    print("methDNA")
    print(f"Sample size: {methDNA.shape[1]}")
    print(f"Feature size: {methDNA.shape[0]}")

    data_mat = [[miRNA.T.values], [mRNA.T.values], [methDNA.T.values]]

    Ariel = entry_point()
    Ariel.set_data_matrix(
        data_mat, 
        likelihoods=['gaussian', 'gaussian', 'gaussian'], 
        views_names=['miRNA', 'mRNA', 'methDNA'],
        features_names=[miRNA.index, mRNA.index, methDNA.index],
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
                'summary': {method: {} for method in result_pack.keys()},
            }
            for method in result_pack.keys():
                data_pack[method] = result_pack[method]
                
                summary_df = pd.DataFrame.from_dict(result_pack[method], orient='index')
                for metric in summary_df.columns:
                    if str(metric).isupper():
                        # Assume all metrics are upper case-noted columns
                        data_pack['summary'][method].update({
                            f'Mean {metric}'    : float(summary_df[metric].mean()),
                            f'Median {metric}'  : float(summary_df[metric].median()),
                            f'Std {metric}'     : float(summary_df[metric].std()),
                            f'Max {metric}'     : float(summary_df[metric].max()),
                            f'Min {metric}'     : float(summary_df[metric].min()),
                        })

                    if str(metric) == 'AUROC':
                        mlflow.log_metric(f'{target_id} {method} Mean AUC', data_pack['summary'][method][f'Mean {metric}'])
                    if str(metric) == 'MCC':
                        mlflow.log_metric(f'{target_id} {method} Mean MCC', data_pack['summary'][method][f'Mean {metric}'])
                    

        
            # Save to MongoDB
            collection.update_one(
                {'run_id': run_id, 'target_id': target_id},
                {'$set': data_pack},
                upsert=True
            )