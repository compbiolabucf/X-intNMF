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


            ablation_data = mongo_db['ABLATION_STUDIES'].find(
                {
                    'dataset': dataset_choice,
                    'target_id': target,
                }
            ).to_list()
            for idx, ablation in enumerate(ablation_data):
                if ablation['run_name'] == 'zero_alpha': ablation_data[idx]['run_name'] = 'X-intMF ($\\alpha$=0)'
                if ablation['run_name'] == 'max_alpha': ablation_data[idx]['run_name'] = 'X-intMF ($\\alpha$=10000)'

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
            
            # st.dataframe(data_to_agg, use_container_width=True)


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

        code_str = agged_result.to_latex(escape=False)
        st.code(code_str, language='latex')



with tab2:
    col1, col2, col3 = st.columns([1, 1, 1])
    omics_choice = col1.selectbox("Omics choice SA", ['2-omic', '3-omic'], placeholder="Select omics choice")
    dataset_choice = col2.selectbox("Dataset SA", ["BRCA", "LUAD", "OV"], placeholder="Select dataset")
    target_list = col3.multiselect("Targets SA", ['Survival', 'Diseasefree'], placeholder="Select targets")
    target_list = [target.lower() for target in target_list]

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
                    },
                    {
                        '_id': 0,
                        'best_cfg_from': 1,
                        'best_cfg': 1,
                        'test_prognostic_index': 1,
                        'p_value': 1,
                        'kaplan_meier_X_low': 1,
                        'kaplan_meier_Y_low': 1,
                        'kaplan_meier_X_high': 1,
                        'kaplan_meier_Y_high': 1,
                        'kaplan_meier_censor_low': 1,
                        'kaplan_meier_censor_high': 1,
                        'kaplan_meier_censor_low_pred': 1,
                        'kaplan_meier_censor_high_pred': 1,
                        'test_low_risk_ids': 1,
                        'test_high_risk_ids': 1,
                        'best_alpha': 1,
                    }
                )
            )
            st.dataframe(data, use_container_width=True)


            for idx in data.index:
                st.markdown(f'##### Best config from {data.loc[idx, "best_cfg_from"]} - {data.loc[idx, "best_cfg"]}')
                st.markdown(f'- Prognostic Index: **{data.loc[idx, "test_prognostic_index"]}**\n- p-value: **{data.loc[idx, "p_value"]}**\n- Alpha: **{data.loc[idx, "best_alpha"]}**')

                Ariel = data.loc[idx]

                X_low = Ariel['kaplan_meier_X_low']
                Y_low = Ariel['kaplan_meier_Y_low']
                X_high = Ariel['kaplan_meier_X_high']
                Y_high = Ariel['kaplan_meier_Y_high']
                best_alpha = Ariel['best_alpha']
                censor_low = Ariel['kaplan_meier_censor_low']
                censor_high = Ariel['kaplan_meier_censor_high']
                censor_low_pred = Ariel['kaplan_meier_censor_low_pred']
                censor_high_pred = Ariel['kaplan_meier_censor_high_pred']
                low_risk_ids = Ariel['test_low_risk_ids']
                high_risk_ids = Ariel['test_high_risk_ids']
                p_value = Ariel['p_value']


                # Plot survival functions using Plotly
                fig = px.line()

                fig.add_scatter(x=X_low, y=Y_low, mode='lines', name=f"low-risk ({len(low_risk_ids)})", line=dict(color='blue', dash='dash'))
                fig.add_scatter(x=X_high, y=Y_high, mode='lines', name=f"high-risk ({len(high_risk_ids)})", line=dict(color='red'))

                # Plot censor points
                fig.add_scatter(x=censor_high, y=censor_high_pred, mode='markers', name='Censor High', marker=dict(color='black', symbol='cross'))
                fig.add_scatter(x=censor_low, y=censor_low_pred, mode='markers', name='Censor Low', marker=dict(color='black', symbol='cross'))

                # Add labels and legend
                fig.update_layout(
                    title="Kaplan-Meier Curves",
                    xaxis_title="Time (Months)",
                    yaxis_title="Survival Probability",
                    legend_title="Risk Group"
                )

                # Add p-value in a box on the bottom left of the plot
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=0.03, y=0.05,
                    text=f'p-value: {p_value:.4f}',
                    showarrow=False,
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=4,
                    bgcolor="white",
                    opacity=1
                )
                fig.update_yaxes(range=[-0.02, 1.02])
                fig.update_xaxes(range=[-0.02, 122]) # max 10 years

                # Show plot
                baseline = data.get('baseline', None)
                if baseline is None:
                    st.plotly_chart(fig)
                else:
                    col1, col2 = st.columns([1, 1])

                    fig_baseline = px.line()
                    fig_baseline.add_scatter(x=baseline['kaplan_meier_X_low'], y=baseline['kaplan_meier_Y_low'], mode='lines', name=f"low-risk ({len(low_risk_ids)})", line=dict(color='blue', dash='dash'))
                    fig_baseline.add_scatter(x=baseline['kaplan_meier_X_high'], y=baseline['kaplan_meier_Y_high'], mode='lines', name=f"high-risk ({len(high_risk_ids)})", line=dict(color='red'))
                    fig_baseline.add_scatter(x=baseline['kaplan_meier_censor_high'], y=baseline['kaplan_meier_censor_high_pred'], mode='markers', name='Censor High', marker=dict(color='black', symbol='cross'))
                    fig_baseline.add_scatter(x=baseline['kaplan_meier_censor_low'], y=baseline['kaplan_meier_censor_low_pred'], mode='markers', name='Censor Low', marker=dict(color='black', symbol='cross'))
                    fig_baseline.update_layout(
                        title="Kaplan-Meier Curves (Baseline)",
                        xaxis_title="Time (Months)",
                        yaxis_title="Survival Probability",
                        legend_title="Risk Group"
                    )

                    fig_baseline.add_annotation(
                        xref="paper", yref="paper",
                        x=0.03, y=0.05,
                        text=f'p-value: {baseline["p_value"]:.4f}',
                        showarrow=False,
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=4,
                        bgcolor="white",
                        opacity=1
                    )
                    fig_baseline.update_yaxes(range=[-0.02, 1.02])
                    fig_baseline.update_xaxes(range=[-0.02, 122])
                    col1.plotly_chart(fig)
                    col2.plotly_chart(fig_baseline)
                    st.markdown(f"Baseline p-value: {baseline['p_value']:.4f}")
                    st.markdown(f"Baseline Alpha: {baseline['best_alpha']:.4f}")

                


                if st.button("Re-train and Download PDF", use_container_width=True, key=uuid.uuid4()):
                    pass













    