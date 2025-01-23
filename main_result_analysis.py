# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: main.py
# Date: 2024/12/10 16:16:08
# Description: 
# 
# (c) 2024 bu1th4nh. All rights reserved. 
# Written with dedication in the University of Central Florida, EPCOT and the Magic Kingdom.
# -----------------------------------------------------------------------------------------------


import mlflow
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Union, Literal

import uuid
import pymongo
import streamlit as st
import plotly.express as px

# -----------------------------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------------------------
st.set_page_config(
    page_title = "Result Analysis",
    page_icon = "⚔️",
    layout = "wide"
)
st.title('Result Analysis | Similar Sample Cross-Omic NMF')


# -----------------------------------------------------------------------------------------------
# MongoDB Connection
# -----------------------------------------------------------------------------------------------
mongo = pymongo.MongoClient(
    host='mongodb://localhost',
    port=27017,
    username='bu1th4nh',
    password='ariel.anna.elsa',
)
mongo_db = mongo['SimilarSampleCrossOmicNMF']



# -----------------------------------------------------------------------------------------------
# Target decoder
# -----------------------------------------------------------------------------------------------
def decode_target(target):
    # Print Target
    if 'survival' in target or 'diseasefree' in target:
        lwr_bound = target.split('_')[1]
        upr_bound = target.split('_')[2]
        classif = target.split('_')[0]
        return f"## {classif.capitalize()}: Under {lwr_bound} or over {upr_bound} months"
    else: return f"## {target}"


# -----------------------------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------------------------
with st.spinner("Loading Data..."):
    for dataset in ['BRCA', 'LUAD', 'OV']:
        if st.session_state.get(dataset, None) is None:
            print(f"Loading {dataset}...")
            st.session_state[dataset] = pd.DataFrame(list(mongo_db[dataset].find({"run_name": {"$regex" : "baseline|overall"}}, {"_id": 0, "run_id": 1, "target_id": 1, "run_name": 1, "summary": 1})))
        else:
            st.session_state[dataset] = pd.DataFrame(list(mongo_db[dataset].find({"run_name": {"$regex" : "baseline|overall"}}, {"_id": 0, "run_id": 1, "target_id": 1, "run_name": 1, "summary": 1})))
            print(f"Already loaded {dataset}.")

    st.session_state['targ_choice'] = {
        'BRCA': ['ER', 'HER2', 'PR', 'TN'],
        # 'LUAD': ['Survival', 'Diseasefree'],
        # 'OV': ['Survival', 'Diseasefree'],
        'LUAD': ['survival_24_30', 'diseasefree_24_24', 'diseasefree_12_18'],
        'OV': ['survival_18_18', 'diseasefree_12_18', 'diseasefree_12_12'],
    }
    

    

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
dataset_choice = col1.selectbox("Dataset", ["BRCA", "LUAD", "OV"], placeholder="Select dataset")


target_list = st.session_state['targ_choice'][dataset_choice]
target_choice = col2.multiselect("Targets", target_list)

metrics = col3.multiselect("Metrics", ['ACC', 'PRE', 'REC', 'F1', 'MCC', 'AUROC', 'AUPRC'], default=['MCC', 'AUROC'])
statistics = col4.multiselect("Statistics", ["Mean", "Median", "Std", "Min", "Max"], default=['Mean'], placeholder="Select statistic")





if len(metrics) == 0: st.stop()
if len(statistics) == 0: st.stop()


if st.button("Retrieve Result", use_container_width=True):
    tab1, tab2, tab3 = st.tabs(["Score", "Strategy 2 Details", "Strategy 1 Details"])

    with tab1:
        data = st.session_state[dataset_choice]
        if data.empty:
            st.warning("No data found.")
            st.stop()
        non_classifier_list = ["run_id", "target_id", "run_name", "summary"]

        # st.write("Label Choices:" + str(target_choice))
        for target in target_choice:
            st.write(decode_target(target))
            
            if dataset_choice == 'BRCA': subdata = data[data["target_id"] == target].copy(deep=True)
            else: subdata = data[data["target_id"].str.contains(target.lower())].copy(deep=True)

            # Process
            finaldata = []
            for index in tqdm(subdata.index, desc="Processing", leave=False):
                
                summary = subdata.loc[index, 'summary']
                run_name = subdata.loc[index, 'run_name']

                # if(run_name == 'overall_our_model'): continue
                if('baseline' in run_name and 'MOFA2' not in run_name): summary = {run_name.split('_')[1]: summary}
                if('overall' in run_name and 'MOFA2' not in run_name): 
                    summary = {"Ours": summary}
                    if subdata.loc[index, 'run_name'] == 'overall_our_model': continue
                    if subdata.loc[index, 'run_name'] == 'overall_our_model_fixed_config': subdata.loc[index, 'run_name'] = 'Our model, strategy 2'
                if('alpha' in run_name): continue

                for classifier in summary.keys():
                    Ariel = summary[classifier]
                    data_row = {
                        'Run Name': subdata.loc[index, 'run_name'].split('_')[-1] if '_' in subdata.loc[index, 'run_name'] else subdata.loc[index, 'run_name'],
                        'Classifier': classifier,
                    }
                    if dataset_choice != 'BRCA':
                        data_row.update({
                            'Lower-bound Threshold': subdata.loc[index, 'target_id'].split('_')[1],
                            'Upper-bound Threshold': subdata.loc[index, 'target_id'].split('_')[2],
                        })

                    for metric in metrics:
                        for stat in statistics:
                            for Belle in Ariel.keys():
                                if metric.lower() in Belle.lower() and stat.lower() in Belle.lower():
                                    data_row[f"{metric} {stat}"] = Ariel[Belle]
                    finaldata.append(data_row)
            
            finaldata = pd.DataFrame(finaldata)
            finaldata.rename(columns={'index': 'test_id'}, inplace=True)
            finaldata.sort_values(by=['AUROC Mean', 'MCC Mean'], inplace=True, ascending=False)


            
            st.dataframe(finaldata, use_container_width=True)

    with tab2:
        data = st.session_state[dataset_choice]
        if data.empty:
            st.warning("No data found.")
            st.stop()
        non_classifier_list = ["run_id", "target_id", "run_name", "summary"]

        # st.write("Label Choices:" + str(target_choice))
        for target in target_choice:
            st.write(decode_target(target))
            
            if dataset_choice == 'BRCA': subdata = data[data["target_id"] == target].copy(deep=True)
            else: subdata = data[data["target_id"].str.contains(target.lower())].copy(deep=True)

            run_strat_detail = mongo_db[dataset_choice].find_one(
                {
                    "run_name": "overall_our_model_fixed_config",
                    "target_id": target,
                }
            )
            hparams_run_detail = list(mongo_db['HPARAMS_OPTS'].find(
                {
                    "dataset": dataset_choice,
                    "target_id": target,
                    "config": run_strat_detail['best_config'],
                    "classifier": run_strat_detail['classifier'],
                },
                {
                    "_id": 0,
                    "test_id": 1,
                    "AUROC": 1,
                }
            ))

            st.markdown(f"**Best Config**: {run_strat_detail['best_config']}")
            st.markdown(f"**Classifier**: {run_strat_detail['classifier']}")
            st.markdown(f"Mean AUROC Train: {run_strat_detail['best_AUROC_CV_train']:.04}")


            test_result = pd.DataFrame.from_dict(run_strat_detail['Overall'], orient='index')[['MCC', 'AUROC']].rename(columns={'MCC': 'Test MCC', 'AUROC': 'Test AUROC'})
            train_cv = pd.DataFrame.from_records(hparams_run_detail).set_index('test_id').rename(columns={'AUROC': 'Train CV AUROC'})

            test_result = test_result.merge(train_cv, how='left', left_index=True, right_index=True)


            col11, col12 = st.columns([1, 3])
            
            formatted_test_result = test_result.style.format({"Train CV AUROC": "{:.4f}".format, "Test AUROC": "{:.4f}".format, })
            col11.dataframe(formatted_test_result, use_container_width=True)


            fig = px.line(
                test_result, 
                x = test_result.index,
                y = ['Test AUROC', 'Train CV AUROC'],
                title=f"Train CV and Test Metrics for {target} in {dataset_choice}", 
                labels={'value': 'Metrics', 'index': 'Test ID'},
                color_discrete_map={'Test AUROC': 'red', 'Train CV AUROC': 'blue'}
            )
            fig.update_yaxes(range=[0, 1])
            col12.plotly_chart(fig,  use_container_width=True, key = str(uuid.uuid4()))

                

    with tab3:
        data = st.session_state[dataset_choice]
        if data.empty:
            st.warning("No data found.")
            st.stop()
        non_classifier_list = ["run_id", "target_id", "run_name", "summary"]

        # st.write("Label Choices:" + str(target_choice))
        for target in target_choice:
            st.write(decode_target(target))
            
            if dataset_choice == 'BRCA': subdata = data[data["target_id"] == target].copy(deep=True)
            else: subdata = data[data["target_id"].str.contains(target.lower())].copy(deep=True)

            run_strat_detail = (
                pd.DataFrame.from_dict(mongo_db[dataset_choice].find_one(
                    {
                        "run_name": "overall_our_model",
                        "target_id": target,
                    }
                )['Overall'], orient='index')
                [['config_id', 'classifier', 'best_AUROC_param_optimization', 'AUROC', 'train_positive_count',  'train_negative_count', 'test_positive_count',  'test_negative_count']]
                .rename(columns={'best_AUROC_param_optimization': 'Train AUROC', 'config_id': 'Config', 'classifier': 'Classifier', 'AUROC': 'Test AUROC'})
            )

            run_strat_detail['Train Cnt'] = run_strat_detail[['train_positive_count',  'train_negative_count']].apply(lambda x: f"{x[0]}/{x[1]}", axis=1)
            run_strat_detail['Test Cnt'] = run_strat_detail[['test_positive_count',  'test_negative_count']].apply(lambda x: f"{x[0]}/{x[1]}", axis=1)
            run_strat_detail = run_strat_detail[['Config', 'Classifier', 'Train AUROC', 'Test AUROC', 'Train Cnt', 'Test Cnt']]

            col21, col22 = st.columns([1, 3])
            col21.dataframe(run_strat_detail, use_container_width=True)

            fig = px.line(
                test_result, 
                x = test_result.index,
                y = ['Test AUROC', 'Train CV AUROC'],
                title=f"Train CV and Test Metrics for {target} in {dataset_choice}", 
                labels={'value': 'Metrics', 'index': 'Test ID'},
                color_discrete_map={'Test AUROC': 'red', 'Train CV AUROC': 'blue'}
            )
            fig.update_yaxes(range=[0, 1])
            col22.plotly_chart(fig,  use_container_width=True, key = str(uuid.uuid4()))



        






