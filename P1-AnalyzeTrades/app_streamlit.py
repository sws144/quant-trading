"""
Analyze Trades Streamlit App - see README.md for details
"""

import importlib
import json
import mlflow
import numpy as np
import pandas as pd
import pickle
import streamlit as st

analyze_pred = importlib.import_module("P1-AnalyzeTrades_h_predictresult")

### SELECTED MODEL ###
st.session_state.runid = "76186ad6e3c543d481ce7508751d91f7"

# load model if not already loaded
if "mdl" not in st.session_state:

    ### load model, cat_dict
    st.session_state.mdl, st.session_state.cat_dict = analyze_pred.preload_model(
        mlflow_tracking_uri="",
        experiment_name="P1-AnalyzeTrades_f_core",
        run_id=runid,
    )

    (
        st.session_state.metrics,
        st.session_state.params,
        st.session_state.tags,
    ) = analyze_pred.parse_mlflow_info(mlflow.get_run(runid))

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
        mlflow_tracking_uri="",
        experiment_name="P1-AnalyzeTrades_f_core",
        run_id=st.session_state.runid,
        inputs=input_df,
        explain=True,
        show_plot=False,
        preloaded_model=st.session_state.mdl,
        categorical_colname_list=list(st.session_state.cat_dict.keys()),
    )

    prediction = np.round(res_df.iloc[0, 0], 3)

    st.image(f"output/current_force.png")


githublink = "[GitHub link](https://github.com/sws144/quant-trading/tree/master/P1-AnalyzeTrades)"
st.markdown(githublink, unsafe_allow_html=False)

# doclink = "[Documentation link](https://analyze-trades-prod.herokuapp.com/doc)"
# st.markdown(doclink, unsafe_allow_html=False)