## flask app file
# type "flask run" in app.py's directory to run

import pandas as pd
from flask import Flask, jsonify, request, render_template, url_for
import pickle
import importlib
import numpy as np
import mlflow
import json

analyze_pred = importlib.import_module("P1-AnalyzeTrades_h_predictresult")

### SELECTED MODEL ###
runid = "efe6aaa6bbaa4fc9ab3d3a36e3a0dacb"

### load model, cat_dict
mdl, cat_dict = analyze_pred.preload_model(
    mlflow_tracking_uri="",
    experiment_name="P1-AnalyzeTrades_f_core",
    run_id=runid,
)

metrics, params, tags = analyze_pred.parse_mlflow_info(mlflow.get_run(runid))

# pull information
col_type_dict = (
    pd.DataFrame(
        json.loads(
            json.loads(tags["mlflow.log-model.history"])[0]["signature"]["inputs"]
        )
    )
    .set_index("name")
    .to_dict(orient="index")
)

version = ""
if "version" in tags.keys():
    version = tags["version"]

## app start
app = Flask(
    __name__, template_folder="templates", static_url_path="", static_folder="output"
)

## routes
@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "GET":
        return render_template(
            "main.html",
            cat_dict=cat_dict,
            col_type_dict=col_type_dict,
            version=version,
        )
    if request.method == "POST":
        # should match main.html form

        # QA
        # print(request.form)

        input_df = pd.DataFrame(request.form, index=[0])

        # CLOSE_VIX = request.form["Q('CLOSE_^VIX')"]
        # AAII_SENT_BULLBEARSPREAD = request.form["Q('AAII_SENT_BULLBEARSPREAD')"]
        # YEARS_TO_NORMALIZATION = request.form["Q('YEARS_TO_NORMALIZATION')"]
        # IMPLIED_P_E = request.form["Q('IMPLIED_P_E')"]

        # inputs = pd.DataFrame(
        #     [[CLOSE_VIX, AAII_SENT_BULLBEARSPREAD, YEARS_TO_NORMALIZATION, IMPLIED_P_E]],
        #     columns=["Q('CLOSE_^VIX')", "Q('AAII_SENT_BULLBEARSPREAD')",
        #              "Q('YEARS_TO_NORMALIZATION')","Q('IMPLIED_P_E')"],
        #     dtype=float)

        # QA
        # print(input_df)

        res_df, shap_obj, shap_df, f = analyze_pred.predict_return(
            mlflow_tracking_uri="",
            experiment_name="P1-AnalyzeTrades_f_core",
            run_id=runid,
            inputs=input_df,
            explain=True,
            show_plot=False,
            preloaded_model=mdl,
            categorical_colname_list=list(cat_dict.keys()),
        )

        prediction = res_df.iloc[0, 0]

        return render_template(
            "main.html",
            cat_dict=cat_dict,
            col_type_dict=col_type_dict,
            version=version,
            original_input=request.form,
            result=str(np.round(prediction, 3)),
        )


@app.route("/doc", methods=["GET"])
def doc():
    if request.method == "GET":
        return render_template(
            "doc.html"
        )  # need to update this with every version update


# def predict():
#     # get data
#     data = request.get_json(force = True)

#     # convert to dataframe
#     data.update((x,[y]) for x , y in data.items())
#     data_df = pd.DataFrame.from_dict(data)

#     # predictions
#     result = model.predict(data_df)

#     # send back to browser
#     output  = {'results': int(result[0])}

#     # return data
#     return jsonify(result=output)

if __name__ == "__main__":
    app.run()
