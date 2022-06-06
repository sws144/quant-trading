# function for predicting one result
import pandas as pd
import numpy as np
import mlflow
import h2o
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import copy
import dill
import json
from os.path import exists
import shap
from shap.plots._waterfall import waterfall_legacy

shap.initjs()  # for plots


def parse_mlflow_info(run_info):
    metrics = run_info.data.metrics
    params = run_info.data.params
    tags = run_info.data.tags
    return metrics, params, tags


def get_model_type(
    tags,
):
    if "estimator_class" in tags.keys():
        if "sklearn" in tags["estimator_class"].lower():
            model_type = "sklearn"
        else:
            model_type = "h2o"
    elif "mlflow.runName" in tags.keys():
        if "h2o" in tags["mlflow.runName"].lower():
            model_type = "h2o"
        else:
            model_type = "h2o"
    else:
        model_type = "h2o"

    mlflow.end_run()

    return model_type


def preload_model(
    mlflow_tracking_uri: str,
    experiment_name: str,
    run_id: str,
):
    """pulls model and cat_dict if available from mlflow

    Args:
        mlflow_tracking_uri (str):
        experiment_name (str):
        run_id (str):

    Returns:
        model , cat_dict:
    """

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    experiment_details = mlflow.get_experiment_by_name(experiment_name)

    mlflow.end_run()
    mlflow.start_run(run_id=run_id)

    # pull model from tracking uri
    artifact_loc = (
        str(experiment_details.artifact_location)
        .replace("file:", "")
        .replace("///", "")
    )

    metrics, params, tags = parse_mlflow_info(mlflow.get_run(run_id))

    model_type = get_model_type(tags)

    if model_type == "sklearn":
        try:  # first try local path]
            mdl = pickle.load(
                open(f"{artifact_loc}/{run_id}/artifacts/model/model.pkl", "rb")
            )
        except:  # then try repo specific path for finalized cases
            mdl = pickle.load(
                open(f"mlruns/0/{run_id}/artifacts/model/model.pkl", "rb")
            )
    else:
        # for h2o models
        h2o.init()
        try:
            logged_model = f"runs:/{run_id}/model"
            # logged_model = f'mlruns/0/{run_id}/artifacts/model'
            mdl = mlflow.pyfunc.load_model(logged_model)

            # mojo deprecated
            # mdl = h2o.import_mojo(f'{artifact_loc}/{run_id}/artifacts/')
        except:
            logged_model = f"mlruns/0/{run_id}/artifacts/model"
            mdl = mlflow.pyfunc.load_model(logged_model)

        mlflow.end_run()

    # load cat dict, if available
    cat_dict = {}
    try:  # first try local path
        cat_dict_loc = f"{artifact_loc}/{run_id}/artifacts/cat_dict.pkl"
        if exists(cat_dict_loc):
            cat_dict = pickle.load(open(cat_dict_loc, "rb"))
    except:  # then try repo specific path for finalized cases
        cat_dict_loc = f"mlruns/0/{run_id}/artifacts/cat_dict.pkl"
        if exists(cat_dict_loc):
            cat_dict = pickle.load(open(cat_dict_loc, "rb"))

    return mdl, cat_dict


def predict_return(
    mlflow_tracking_uri: str,
    experiment_name: str,
    run_id: str,
    inputs: pd.DataFrame,
    explain: bool = False,
    show_plot: bool = False,
    preloaded_model=None,
    categorical_colname_list=None,
):
    """Predict the return of model in decimal form

    Args:
        mlflow_tracking_uri (str): where mlflow runs sit

        experiment_name (str): runs are organized into experimetns

        run_id (str): specific run

        inputs (pd.DataFrame): raw dataframe (can be multiple rows)

        explain (bool = False): explain results using shap

        show_plot (bool = False): show plot when running interactively. MAY NOT BE WORKING

        preloaded_model (obj = None): can send preloaded model for speed

        categorical_colname_list (list = None): can send list of strings for columns
            to force to categorical

    Returns:
        pct_return: dataframe of results

        (if explain == True, show force plot of explained results and return below)
        shap_obj: a shap values object multiple attributes (including .values)
        shap_df: a df of shap values
        f: figure
    """

    plt.clf()  # clear current figure

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    experiment_details = mlflow.get_experiment_by_name(experiment_name)

    mlflow.end_run()
    mlflow.start_run(run_id=run_id)

    # pull model from tracking uri
    artifact_loc = (
        str(experiment_details.artifact_location)
        .replace("file:", "")
        .replace("///", "")
    )

    metrics, params, tags = parse_mlflow_info(mlflow.get_run(run_id))

    # add columns if necessary, can only add, not remove extra cols
    cols_required = list(
        pd.DataFrame(
            json.loads(
                json.loads(tags["mlflow.log-model.history"])[0]["signature"]["inputs"]
            )
        )["name"]
    )

    # todo replace above
    col_type_dict = (
        pd.DataFrame(
            json.loads(
                json.loads(tags["mlflow.log-model.history"])[0]["signature"]["inputs"]
            )
        )
        .set_index("name")
        .to_dict(orient="index")
    )

    # ensure categorical splits contained necessary columns
    add_cols = list(set(cols_required) - set(list(inputs.columns)))
    inputs_copy = inputs.copy()
    inputs_copy[add_cols] = 0

    # extra columns in dataset
    # print('extra columns in expanded dataset: '+  str(list(set(list(inputs_copy.columns)) - set(cols_required))))

    # ensure X is in correct order and complete for model
    inputs_copy = inputs_copy[cols_required]

    for c in inputs_copy.columns:
        if col_type_dict[c]["type"] == "double":
            inputs_copy[c] = inputs_copy[c].astype(float)

    if preloaded_model == None:
        mdl, _ = preload_model(
            mlflow_tracking_uri,
            experiment_name,
            run_id,
        )
    else:
        mdl = preloaded_model

    mlflow.end_run()

    # consider later
    # formula_clean = params['formula'].replace('\n','')

    model_type = get_model_type(tags)

    if model_type == "sklearn":
        pct_return = mdl.predict(inputs_copy)
        pct_return_df = pd.DataFrame(pct_return, columns=["predicted_ret"])
    else:
        # assume H2O
        pct_return = mdl.predict(inputs_copy)
        pct_return_df = pct_return.rename(columns={"predict": "predicted_ret"})

    # Explain Return for first
    if explain == True:
        try:
            explainer = dill.load(
                open(f"{artifact_loc}/{run_id}/artifacts/explainer.pkl", "rb")
            )
        except:  # for testing
            explainer = dill.load(
                open(f"mlruns/0/{run_id}/artifacts/explainer.pkl", "rb")
            )

        # create explained object
        if "pipeline" in str(type(mdl)):
            ## fix shap_obj, requires column transformer in step position 0 ,
            ## categorical in position 1
            shap_obj = explainer(mdl[0].transform(inputs_copy))
        else:
            shap_obj = explainer(inputs_copy)

        # correct for error
        shap_obj.base_values = shap_obj.base_values

        # shap values df with column
        shap_df = pd.DataFrame(shap_obj.values, columns=inputs_copy.columns)

        # ensure pct return matches shap, in case of gbm explanation of linear
        adj = (
            pct_return_df.sum().sum()
            - shap_df.sum().sum()
            - float(shap_obj.base_values[0])
        )
        specific_adj = adj / shap_df.shape[1]

        if abs(adj) > 0.01:
            print("warning, adjusting shap to match actual")
        shap_df = shap_df + specific_adj

        # shap.plots.force(shap_obj.base_values[0][0],
        #                     shap_values = shap_obj.values,
        #                     features = inputs_copy.columns,
        #                     matplotlib = True,
        #                     show = False)

        try:
            shap_obj_adj = shap_obj
            shap_obj_adj.values = shap_obj_adj.values + specific_adj

            if "pipeline" in str(type(mdl)):
                # def update_shap_obj(shap_obj, X_train, encoder):
                shap_obj_adj.feature_names = list(inputs_copy.columns)

                if categorical_colname_list is None:
                    categorical_names = list(
                        inputs_copy.select_dtypes(include=["object"]).columns
                    )
                else:
                    categorical_names = categorical_colname_list
                col_idx = list(
                    np.where(np.isin(shap_obj_adj.feature_names, categorical_names))[0]
                )

                shap_cat = copy.deepcopy(shap_obj_adj)
                shap_cat.data = np.array(shap_obj_adj.data, dtype="object")
                res_arr = (
                    mdl[0]
                    .transformers_[1][1][1]
                    .inverse_transform(
                        pd.DataFrame(
                            shap_cat.data[:, col_idx], columns=[categorical_names]
                        )
                    )
                )
                for i, loc in enumerate(col_idx):
                    shap_cat.data[:, loc] = res_arr[:, i]

                shap.plots.waterfall(shap_cat[0])
            else:
                shap.plots.waterfall(shap_obj_adj[0])
            # waterfall_legacy(shap_obj.base_values[0][0],
            #                     shap_values = shap_obj.values.ravel()+specific_adj,
            #                     feature_names = inputs_copy.columns,
            #                     show = False)
        except:
            # backup, probably not working
            print(f"backup, probably  not working")
            waterfall_legacy(
                shap_obj.base_values[0],
                shap_values=shap_obj.values.ravel() + specific_adj,
                feature_names=inputs_copy.columns,
                show=False,
            )

        f = plt.gcf()
        f.tight_layout()
        f.savefig("output/current_force.png")
        if show_plot:
            # matplotlib.use('svg')
            plt.show()

        return pct_return_df, shap_obj, shap_df, f

    else:
        return pct_return_df
