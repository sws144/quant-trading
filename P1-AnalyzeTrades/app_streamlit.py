"""
Analyze Trades Streamlit App - see README.md for details
"""

import importlib
import json
import mlflow
import numpy as np
import pandas as pd
import os
import pickle
import streamlit as st

analyze_pred = importlib.import_module("P1-AnalyzeTrades_h_predictresult")

### SELECTED MODEL ###
st.session_state.runid = "8f46b4f8125245d9a94d68066d2f051d"

current_uri = os.getcwd()
# QA for tracking_uri
# st.write(current_uri)

tracking_uri = ""
# # streamlit uses github root dir, so need to go into folder if not available
if "P1-AnalyzeTrades" not in current_uri:
    # insertloc = current_uri.rfind(r"/")
    tracking_uri = current_uri + "/P1-AnalyzeTrades/mlruns"
    # QA
    # st.write(tracking_uri)

# load model if not already loaded
if "mdl" not in st.session_state:

    ### load model, cat_dict
    st.session_state.mdl, st.session_state.cat_dict = analyze_pred.preload_model(
        mlflow_tracking_uri=tracking_uri,
        experiment_name="P1-AnalyzeTrades_f_core",
        run_id=st.session_state.runid,
    )

    # QA
    # st.write(st.session_state.cat_dict)
    # st.write(mlflow.get_tracking_uri())

    (
        st.session_state.metrics,
        st.session_state.params,
        st.session_state.tags,
    ) = analyze_pred.parse_mlflow_info(mlflow.get_run(st.session_state.runid))

    # pull information
    st.session_state.col_type_dict = (
        pd.DataFrame(
            json.loads(
                json.loads(st.session_state.tags["mlflow.log-model.history"])[0][
                    "signature"
                ]["inputs"]
            )
        )
        .set_index("name")
        .to_dict(orient="index")
    )

    st.session_state.version = ""
    if "version" in st.session_state.tags.keys():
        st.session_state.version = st.session_state.tags["version"]

st.title(f"Analyze Trades App ")
st.write(
    f"This is a sample model to to predict the return of an individual trade given starting characteristics, using machine learning"
)
st.write(f"Version {st.session_state.version} - Stanley W")

st.write(f"Select Inputs at left")

# create input tab
input_dict = {}
with st.sidebar:
    with st.form(key="my_form"):
        st.title("Inputs")
        for variable, vartype in st.session_state.col_type_dict.items():
            if variable in st.session_state.cat_dict.keys():
                input_dict[variable] = st.selectbox(
                    label=variable, options=st.session_state.cat_dict[variable]
                )
            else:
                input_dict[variable] = st.number_input(label=variable)
        submitted = st.form_submit_button(label="Submit")


if submitted:
    input_df = pd.DataFrame(input_dict, index=[0])

    res_df, shap_obj, shap_df, f = analyze_pred.predict_return(
        mlflow_tracking_uri=tracking_uri,
        experiment_name="P1-AnalyzeTrades_f_core",
        run_id=st.session_state.runid,
        inputs=input_df,
        explain=True,
        show_plot=False,
        preloaded_model=st.session_state.mdl,
        categorical_colname_list=list(st.session_state.cat_dict.keys()),
    )

    prediction = np.round(res_df.iloc[0, 0], 3)

    st.image(f"output/current_force.png")  # based on predict_return function output


githublink = "[GitHub link](https://github.com/sws144/quant-trading/tree/master/P1-AnalyzeTrades)"
st.markdown(githublink, unsafe_allow_html=False)

# doclink = "[Documentation link](https://analyze-trades-prod.herokuapp.com/doc)"
# st.markdown(doclink, unsafe_allow_html=False)
