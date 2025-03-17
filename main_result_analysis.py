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


import requests
import mlflow
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Union, Literal

import base64
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
    for omics_choice in ['2-omic', '3-omic']:
        st.session_state[omics_choice] = {}
        db_id = 'SimilarSampleCrossOmicNMF_3Omics' if omics_choice == '3-omic' else 'SimilarSampleCrossOmicNMF'
        mongo_db = mongo[db_id]
        for dataset in ['BRCA', 'LUAD', 'OV']:
            print(f"Loading {dataset} of {omics_choice}...")
            st.session_state[omics_choice][dataset] = pd.DataFrame(list(mongo_db[dataset].find({"run_name": {"$regex" : "baseline|overall"}}, {"_id": 0, "run_id": 1, "target_id": 1, "run_name": 1, "summary": 1})))


    st.session_state['targ_choice'] = {
        'BRCA': ['ER', 'HER2', 'PR', 'TN'],
        # 'LUAD': ['Survival', 'Diseasefree'],
        # 'OV': ['Survival', 'Diseasefree'],
        'LUAD': ['survival_24_30', 'diseasefree_24_24', 'diseasefree_12_18'],
        'OV': ['survival_18_18', 'diseasefree_12_18', 'diseasefree_12_12'],
    }
    

tab1, tab2 = st.tabs(["Classification", "Survival Analysis"])

with tab1:
    col0, col1, col2, col3, col4 = st.columns([1, 1, 1, 1, 1])

    omics_choice = col0.selectbox("Omics choice", ['2-omic', '3-omic'], placeholder="Select omics choice")
    dataset_choice = col1.selectbox("Dataset", ["BRCA", "LUAD", "OV"], placeholder="Select dataset")


    target_list = st.session_state['targ_choice'][dataset_choice]
    target_choice = col2.multiselect("Targets", target_list)


    metrics = col3.multiselect("Metrics", ['ACC', 'PRE', 'REC', 'F1', 'MCC', 'AUROC', 'AUPRC'], default=['MCC', 'AUROC'])
    statistics = col4.multiselect("Statistics", ["Mean", "Median", "Std", "Min", "Max"], default=['Mean'], placeholder="Select statistic")

    show_indiv_targets = st.checkbox("Show individual targets", value=False)

    if len(metrics) == 0: st.stop()
    if len(statistics) == 0: st.stop()
    if st.button("Retrieve Result", use_container_width=True):

        agged_result = None

        data = st.session_state[omics_choice][dataset_choice]
        if data.empty:
            st.warning("No data found.")
            st.stop()
        non_classifier_list = ["run_id", "target_id", "run_name", "summary"]

        # st.write("Label Choices:" + str(target_choice))
        for target in target_choice:
            
            if dataset_choice == 'BRCA': subdata = data[data["target_id"] == target].copy(deep=True)
            else: subdata = data[data["target_id"].str.contains(target.lower())].copy(deep=True)

            # Process
            finaldata = []
            for index in tqdm(subdata.index, desc="Processing", leave=False):
                
                summary = subdata.loc[index, 'summary']
                run_name = subdata.loc[index, 'run_name']
                if('mRNAmethDNA' in run_name or 'miRNAmethDNA' in run_name):
                    Ariel = run_name.split('_')
                    run_name = f"{Ariel[0]}_{Ariel[1]}+{Ariel[2]}"
                    continue



                if('baseline' in run_name and 'MOFA2' not in run_name): summary = {run_name.split('_')[1]: summary}
                if('overall' in run_name and 'MOFA2' not in run_name): 
                    summary = {"Ours": summary}
                    if subdata.loc[index, 'run_name'] == 'overall_our_model': continue
                    if subdata.loc[index, 'run_name'] == 'overall_our_model_fixed_config': subdata.loc[index, 'run_name'] = 'X-intMF'
                if('alpha' in run_name): continue



                for classifier in summary.keys():
                    Ariel = summary[classifier]
                    data_row = {
                        'Run Name': run_name.split('_')[-1] if '_' in subdata.loc[index, 'run_name'] else subdata.loc[index, 'run_name'],
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


            db_id = 'SimilarSampleCrossOmicNMF_3Omics' if omics_choice == '3-omic' else 'SimilarSampleCrossOmicNMF'
            ablation_data = mongo[db_id]['ABLATION_STUDIES'].find(
                {
                    'dataset': dataset_choice,
                    'target_id': target,
                }
            ).to_list()
            for idx, ablation in enumerate(ablation_data):
                if ablation['run_name'] == 'zero_alpha': ablation_data[idx]['run_name'] = 'X-intMF ($\\alpha=0$)'
                if ablation['run_name'] == 'max_alpha': ablation_data[idx]['run_name'] = 'X-intMF ($\\alpha=10000$)'
                if 'best-at' in ablation['run_name']: ablation_data[idx]['run_name'] = f"X-intMF (@{ablation['run_name'].split('-')[-2]})"
                if ablation['run_name'] == 'baseline': ablation_data[idx]['run_name'] = 'Baseline'

            for ablation in ablation_data:
                data_row = {
                    'Run Name': ablation['run_name'],
                    'Classifier': ablation['classifier'],
                }
                for metric in metrics:
                    for stat in statistics:
                        keyword = f"{stat} {metric}"
                        if keyword in ablation['summary'].keys():
                            data_row[f"{metric} {stat}"] = ablation['summary'][keyword]
                finaldata.append(data_row)

            

            
            finaldata = pd.DataFrame(finaldata)
            finaldata.rename(columns={'index': 'test_id'}, inplace=True)
            finaldata.sort_values(by=['AUROC Mean', 'MCC Mean'], inplace=True, ascending=False)

            
            # st.write(decode_target(target))
            # st.dataframe(finaldata.drop(columns=['Lower-bound Threshold', 'Upper-bound Threshold']), use_container_width=True)



            data_to_agg = finaldata[['Run Name', 'MCC Mean', 'AUROC Mean']].rename(columns={'AUROC Mean': f'{target} AUROC', 'MCC Mean': f'{target} MCC'})


            for idx in data_to_agg.index:
                name = finaldata.loc[idx, 'Run Name']
                if '+' in name: data_to_agg.loc[idx, 'Run Name'] = str(name).replace('+', ' + ')
                if 'MOFA2' in name: data_to_agg.loc[idx, 'Run Name'] = f'{name} + {finaldata.loc[idx, "Classifier"]}'



            data_to_agg.set_index('Run Name', inplace=True)
            data_to_agg.sort_index(inplace=True, ascending=False)

            if agged_result is None: agged_result = data_to_agg
            else:
                if agged_result.index.equals(data_to_agg.index):
                    for col in data_to_agg.columns:
                        agged_result[col] = data_to_agg[col]
            
            if show_indiv_targets:
                st.markdown(decode_target(target))
                st.dataframe(data_to_agg, use_container_width=True)


        st.markdown("## Aggregated Result")
        st.dataframe(agged_result, use_container_width=True)

        def format_max_col(column):
            max_val = np.max(column)
            rtn = []
            for x in column:
                if x == max_val: rtn.append(f"\\textbf{{{x:.04f}}}")
                else: rtn.append(f"{x:.04f}")
            return rtn
        
        def format_name(x):
            if ' + Random Forest' in x: return x.replace('2 + Random Forest', '+RF')
            elif ' + Logistic Regression' in x: return x.replace('2 + Logistic Regression', '+LR')
            else: return x
        
        agged_result = agged_result.apply(format_max_col)
        agged_result.index = agged_result.index.map(format_name)

        st.markdown("## LaTeX Code")
        code_str = agged_result.to_latex(escape=False)
        st.code(code_str, language='latex')



with tab2:
    col1, col2, col3 = st.columns([1, 1, 1])
    omics_choice = col1.selectbox("Omics choice SA", ['2-omic', '3-omic'], placeholder="Select omics choice")
    dataset_choice = col2.selectbox("Dataset SA", ["BRCA", "LUAD", "OV"], placeholder="Select dataset")
    target_list = col3.multiselect("Targets SA", ['Survival', 'Diseasefree'], placeholder="Select targets")
    target_list = [target.lower() for target in target_list]

    only_show_best = st.checkbox("Only show best result", value=True)

    if len(target_list) == 0: st.stop()

    if st.button("Retrieve Surv. Result", use_container_width=True):
        for target in target_list:
            st.markdown(f'## {target.capitalize()}')
            
            mongo_db = mongo['SimilarSampleCrossOmicNMF_3Omics'] if omics_choice == '3-omic' else mongo['SimilarSampleCrossOmicNMF']
            collection = mongo_db['SURVIVAL_ANALYSIS']


            # Get data
            data = pd.DataFrame.from_records(
                collection
                .find(
                    {
                        "dataset_id": dataset_choice,
                        "surv_target": target,
                    }
                )
            )
            st.dataframe(data, use_container_width=True)


            for idx in data.index:
                Ariel = data.loc[idx]
                try:
                    Jasmine = requests.post(
                        url='http://localhost:6789/survival_pdf',
                        json={
                            'omics_mode':           omics_choice,
                            'dataset_id':           dataset_choice,
                            'surv_target':          target,
                            'best_cfg':             Ariel['best_cfg'],
                            'best_alpha':           Ariel['best_alpha'],
                            'train_sample_ids':     list(Ariel['train_sample_ids']),
                            'test_sample_ids':      list(Ariel['test_sample_ids']),
                        },
                        # timeout=15
                    )
                except requests.exceptions.RequestException as e:
                    st.error(f"Error: {e}")
                    continue


                
                
                if Jasmine.status_code == 200:
                    # st.success("PDF generated successfully!")
                    pdf_base64 = Jasmine.json()['kmf_plot']
                    baseline_pvalue = Jasmine.json()['baseline_pvalue']
                    ablated_pvalue = Jasmine.json()['ablated_pvalue']
                    model_pvalue = Ariel["p_value"]

                    verdict = None
                    if model_pvalue < baseline_pvalue and model_pvalue < 0.05:
                        if model_pvalue < 0.01 and baseline_pvalue > 0.1:
                            verdict = '<span style="color:green"> **Recommended, best attempt**</span>.'
                        elif baseline_pvalue < 0.05:
                            verdict = '<span style="color:red"> **Not recommended**</span>.'
                        else:
                            verdict = '<span style="color:green"> **Choose**</span>.'
                    else:
                        verdict = '<span style="color:red"> **Not recommended**</span>.'


                    if only_show_best and ('best' not in str(verdict)): continue


                    st.markdown(f'### Best config from {data.loc[idx, "best_cfg_from"]} - {data.loc[idx, "best_cfg"]}')
                    st.markdown(f'''
                        - Prognostic Index: **{data.loc[idx, "test_prognostic_index"]:.06f}**
                        - Alpha: **{Ariel["best_alpha"]:.06f}**
                        - Model p-value: **{Ariel["p_value"]:.06f}**
                        - Ablated p-value: **{ablated_pvalue:.06f}**
                        - Baseline p-value: **{baseline_pvalue:.06f}**
                        - Verdict: {verdict}
                        - PDF preview:
                    ''', unsafe_allow_html=True)
                    # Show this PDF in Streamlit
                    st.markdown(f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="100%" height="475px"></iframe>', unsafe_allow_html=True)
                else:
                    st.error("Failed to generate PDF.")
                    st.error(Jasmine.text)
                













    