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
import io

# Set page config for wider layout
st.set_page_config(
    page_title="Analyze Trades App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS to increase app width
st.markdown(
    """
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
        max-width: 95%;
    }
    
    .stDataFrame {
        font-size: 0.9rem;
    }
    
    .stSelectbox > div > div {
        font-size: 0.9rem;
    }
    
    .stNumberInput > div > div > input {
        font-size: 0.9rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

analyze_pred = importlib.import_module("P1-AnalyzeTrades_h_predictresult")

### SELECTED MODEL ###
st.session_state.runid = "d43ec62077544139b7e105bb275596d3"

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

st.write("---")

# Create main layout with inputs on left and results on right
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("📊 Input Configuration")

    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        [
            "Individual Fields (Table Format)",
            "Bulk Copy/Paste from Excel",
            "Interactive Table",
        ],
        key="input_method",
        horizontal=True,
    )

    st.write("---")

    # create input tab
    input_dict = {}

    with st.form(key="my_form"):
        if input_method == "Individual Fields (Table Format)":
            st.subheader("📝 Individual Input Fields")

            # Split variables between two sub-columns
            sub_col1, sub_col2 = st.columns(2)
            variables = list(st.session_state.col_type_dict.keys())
            mid_point = len(variables) // 2

            with sub_col1:
                for variable in variables[:mid_point]:
                    vartype = st.session_state.col_type_dict[variable]
                    if variable in st.session_state.cat_dict.keys():
                        input_dict[variable] = st.selectbox(
                            label=variable,
                            options=st.session_state.cat_dict[variable],
                            help=f"Valid options: {', '.join(st.session_state.cat_dict[variable])}",
                        )
                    else:
                        input_dict[variable] = st.number_input(
                            label=variable,
                            help=f"Expected type: {vartype.get('type', 'numeric')}",
                        )

            with sub_col2:
                for variable in variables[mid_point:]:
                    vartype = st.session_state.col_type_dict[variable]
                    if variable in st.session_state.cat_dict.keys():
                        input_dict[variable] = st.selectbox(
                            label=variable,
                            options=st.session_state.cat_dict[variable],
                            help=f"Valid options: {', '.join(st.session_state.cat_dict[variable])}",
                        )
                    else:
                        input_dict[variable] = st.number_input(
                            label=variable,
                            help=f"Expected type: {vartype.get('type', 'numeric')}",
                        )

        elif input_method == "Bulk Copy/Paste from Excel":
            st.subheader("📋 Bulk Input from Excel")
            st.markdown(
                """
                **Instructions:**
                1. Copy your data from Excel (select cells and Ctrl+C)
                2. Paste into the text area below
                3. Data should be in this order (one row):
                """
            )

            # Show expected column order
            col_order = list(st.session_state.col_type_dict.keys())
            st.code(" | ".join(col_order), language="text")

            # Show example data
            with st.expander("📋 Example Data Format", expanded=False):
                st.markdown("**Example input (copy this format):**")
                example_values = []
                for variable in col_order:
                    if variable in st.session_state.cat_dict.keys():
                        # Use first valid option as example
                        example_values.append(st.session_state.cat_dict[variable][0])
                    else:
                        # Use reasonable numeric examples
                        if "VIX" in variable:
                            example_values.append("25.5")
                        elif "P_E" in variable:
                            example_values.append("15.2")
                        elif "%" in variable:
                            example_values.append("5.0")
                        else:
                            example_values.append("1.0")

                example_text = "\t".join(example_values)
                st.code(example_text, language="text")
                st.markdown(
                    "*Copy the above line and paste it into the text area below*"
                )

            # Text area for bulk input
            bulk_input = st.text_area(
                "Paste Excel data here (tab-separated):",
                height=100,
                help="Paste your Excel data here. Values should be separated by tabs.",
            )

            # Parse bulk input
            if bulk_input.strip():
                try:
                    # Split by tabs and clean up
                    values = [v.strip() for v in bulk_input.strip().split("\t")]

                    # Validate number of values
                    expected_cols = len(st.session_state.col_type_dict)
                    if len(values) != expected_cols:
                        st.error(f"Expected {expected_cols} values, got {len(values)}")
                        st.stop()

                    # Convert to input_dict with validation
                    for i, (variable, vartype) in enumerate(
                        st.session_state.col_type_dict.items()
                    ):
                        value = values[i]

                        if variable in st.session_state.cat_dict.keys():
                            # Categorical validation
                            valid_options = st.session_state.cat_dict[variable]
                            if value not in valid_options:
                                st.error(
                                    f"Invalid value '{value}' for {variable}. Valid options: {', '.join(valid_options)}"
                                )
                                st.stop()
                            input_dict[variable] = value
                        else:
                            # Numeric validation
                            try:
                                numeric_value = float(value)
                                input_dict[variable] = numeric_value
                            except ValueError:
                                st.error(
                                    f"Invalid numeric value '{value}' for {variable}"
                                )
                                st.stop()

                    st.success("✅ Input parsed successfully!")

                    # Show preview of parsed data
                    with st.expander("👁️ Preview Parsed Data", expanded=True):
                        preview_df = pd.DataFrame([input_dict])
                        st.dataframe(preview_df, use_container_width=True)

                except Exception as e:
                    st.error(f"Error parsing input: {str(e)}")
                    st.stop()
            else:
                st.warning("Please paste your Excel data above")
                st.stop()

        elif input_method == "Interactive Table":
            st.subheader("📊 Interactive Table Input")
            st.markdown(
                """
                **Instructions:**
                1. Use the table below to input your data
                2. Click on cells to edit values directly
                3. Use buttons to manage rows
                """
            )

            # Show expected columns
            col_order = list(st.session_state.col_type_dict.keys())
            st.info(f"**Expected columns:** {', '.join(col_order)}")

            # Initialize session state for DataFrame if not exists
            if "interactive_df" not in st.session_state:
                # Create example DataFrame with proper structure
                example_data = {}
                for variable in col_order:
                    if variable in st.session_state.cat_dict.keys():
                        example_data[variable] = [
                            st.session_state.cat_dict[variable][0]
                        ]
                    else:
                        # Use reasonable numeric examples
                        if "VIX" in variable:
                            example_data[variable] = [25.5]
                        elif "P_E" in variable:
                            example_data[variable] = [15.2]
                        elif "%" in variable:
                            example_data[variable] = [0.0]
                        else:
                            example_data[variable] = [0.0]

                st.session_state.interactive_df = pd.DataFrame(example_data)

            # Initialize edited_df session state if not exists
            if "edited_df" not in st.session_state:
                st.session_state.edited_df = st.session_state.interactive_df.copy()

            # Interactive DataFrame editor
            st.subheader("📝 Edit Data (Click to modify values)")
            st.markdown(
                "*💡 Tip: Click on any cell to edit. Use the buttons below to manage rows.*"
            )

            # Convert to string for editing and transpose for vertical display
            string_df = st.session_state.edited_df.astype(str)
            vertical_df = string_df.T
            vertical_df.columns = [
                f"Row {i+1}" for i in range(len(vertical_df.columns))
            ]

            # Use st.data_editor for interactive editing (vertical format, all strings)
            edited_vertical_df = st.data_editor(
                vertical_df,
                use_container_width=True,
                num_rows="dynamic",
                key="data_editor",
            )

            # Transpose back to original format
            edited_string_df = edited_vertical_df.T
            edited_string_df.columns = st.session_state.interactive_df.columns

            # Convert back to appropriate data types
            edited_df = edited_string_df.copy()
            for col in edited_df.columns:
                if col in st.session_state.cat_dict.keys():
                    # Keep as string for categorical (validation will check)
                    pass
                else:
                    # Convert numeric columns back to numeric
                    try:
                        edited_df[col] = pd.to_numeric(edited_df[col], errors="coerce")
                    except:
                        pass

            # Update session state with edited data immediately (inside form)
            st.session_state.edited_df = edited_df

            # Validate edited data
            validation_errors = []
            for col in edited_df.columns:
                if col in st.session_state.cat_dict.keys():
                    # Categorical validation
                    valid_options = st.session_state.cat_dict[col]
                    invalid_values = edited_df[~edited_df[col].isin(valid_options)][
                        col
                    ].unique()
                    if len(invalid_values) > 0:
                        validation_errors.append(
                            f"{col}: Invalid values {list(invalid_values)}. Valid options: {', '.join(valid_options)}"
                        )
                else:
                    # Numeric validation
                    try:
                        edited_df[col] = pd.to_numeric(edited_df[col], errors="coerce")
                        if edited_df[col].isna().any():
                            validation_errors.append(
                                f"{col}: Contains non-numeric values"
                            )
                    except:
                        validation_errors.append(f"{col}: Cannot convert to numeric")

            if validation_errors:
                st.error("Validation errors:")
                for error in validation_errors:
                    st.error(f"• {error}")
                st.stop()

            # Convert to input_dict format for prediction
            if len(edited_df) == 1:
                input_dict = edited_df.iloc[0].to_dict()
            else:
                st.info(
                    f"Table has {len(edited_df)} rows. Using first row for prediction."
                )
                input_dict = edited_df.iloc[0].to_dict()

            # Show current data summary (vertical format)
            with st.expander("📋 Current Data Summary", expanded=False):
                summary_vertical = edited_df.T
                summary_vertical.columns = [
                    f"Row {i+1}" for i in range(len(summary_vertical.columns))
                ]
                st.dataframe(summary_vertical, use_container_width=True)

        # Submit button
        submitted = st.form_submit_button(
            label="🚀 Run Prediction", use_container_width=True, type="primary"
        )

    # Control buttons for Interactive Table (outside of form)
    if input_method == "Interactive Table":
        st.subheader("🔧 Table Controls")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(
                "🔄 Reset to Example Data", help="Reset to default example values"
            ):
                # Reset to example data
                col_order = list(st.session_state.col_type_dict.keys())
                example_data = {}
                for variable in col_order:
                    if variable in st.session_state.cat_dict.keys():
                        example_data[variable] = [
                            st.session_state.cat_dict[variable][0]
                        ]
                    else:
                        if "VIX" in variable:
                            example_data[variable] = [25.5]
                        elif "P_E" in variable:
                            example_data[variable] = [15.2]
                        elif "%" in variable:
                            example_data[variable] = [5.0]
                        else:
                            example_data[variable] = [1.0]
                st.session_state.interactive_df = pd.DataFrame(example_data)
                st.session_state.edited_df = st.session_state.interactive_df.copy()
                st.rerun()

        with col2:
            if st.button("➕ Add Row", help="Add a new row to the table"):
                # Add a new row with default values
                col_order = list(st.session_state.col_type_dict.keys())
                new_row = {}
                for variable in col_order:
                    if variable in st.session_state.cat_dict.keys():
                        new_row[variable] = st.session_state.cat_dict[variable][0]
                    else:
                        new_row[variable] = 0.0

                new_df = pd.concat(
                    [st.session_state.edited_df, pd.DataFrame([new_row])],
                    ignore_index=True,
                )
                st.session_state.edited_df = new_df
                st.rerun()

        with col3:
            if st.button("🗑️ Clear All", help="Clear all data"):
                col_order = list(st.session_state.col_type_dict.keys())
                st.session_state.edited_df = pd.DataFrame(columns=col_order)
                st.rerun()

with right_col:
    st.subheader("🎯 Prediction Results")

    # Initialize session state for results if not exists
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
        st.session_state.input_df_result = None

    if submitted:
        with st.spinner("🔄 Running prediction..."):
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

            # Store results in session state
            st.session_state.prediction_result = prediction
            st.session_state.input_df_result = input_df

    # Display results
    if st.session_state.prediction_result is not None:
        # Display prediction prominently
        st.metric(
            label="Predicted Return",
            value=f"{st.session_state.prediction_result:.3f}",
            help="Predicted percentage return for the trade",
        )

        # Show SHAP explanation
        st.subheader("📊 Feature Importance (SHAP)")
        st.image(f"output/current_force.png")  # based on predict_return function output

        # Show input summary
        with st.expander("📋 Input Summary", expanded=False):
            st.dataframe(st.session_state.input_df_result, use_container_width=True)
    else:
        st.info("👈 Enter your inputs and click 'Run Prediction' to see results here")


githublink = "[GitHub link](https://github.com/sws144/quant-trading/tree/master/P1-AnalyzeTrades)"
st.markdown(githublink, unsafe_allow_html=False)

# doclink = "[Documentation link](https://analyze-trades-prod.herokuapp.com/doc)"
# st.markdown(doclink, unsafe_allow_html=False)
