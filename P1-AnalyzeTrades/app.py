# flask app file
# type "flask run" in app.py's directory to run

import pandas as pd
from flask import Flask, jsonify, request, render_template, url_for
import pickle
import importlib
import numpy as np
analyze_pred = importlib.import_module("P1-AnalyzeTrades_h_predictresult") 

### SELECTED MODEL ###
runid = '1140c5a2c378445ba06b77647d969345'
######################

### load model
mdl = analyze_pred.preload_model(
    mlflow_tracking_uri = '', 
    experiment_name =  'P1-AnalyzeTrades_f_core', 
    run_id =  runid , 
)


# app
app = Flask(__name__, template_folder='templates', static_url_path='', 
            static_folder='output')

# routes
@app.route('/',methods=['GET','POST'])
def main():
    if request.method == 'GET':
        return(render_template('main.html'))
    if request.method == 'POST':
        # should match main.html form
        CLOSE_VIX = request.form["Q('CLOSE_^VIX')"]
        AAII_SENT_BULLBEARSPREAD = request.form["Q('AAII_SENT_BULLBEARSPREAD')"]
        YEARS_TO_NORMALIZATION = request.form["Q('YEARS_TO_NORMALIZATION')"]
        IMPLIED_P_E = request.form["Q('IMPLIED_P_E')"]        
        
        inputs = pd.DataFrame(
            [[CLOSE_VIX, AAII_SENT_BULLBEARSPREAD, YEARS_TO_NORMALIZATION, IMPLIED_P_E]],
            columns=["Q('CLOSE_^VIX')", "Q('AAII_SENT_BULLBEARSPREAD')",
                     "Q('YEARS_TO_NORMALIZATION')","Q('IMPLIED_P_E')"],
            dtype=float)        

        res_df, shap_obj, shap_df, f = analyze_pred.predict_return(
            mlflow_tracking_uri = '', 
            experiment_name =  'P1-AnalyzeTrades_f_core', 
            run_id =  runid, 
            inputs = inputs, 
            explain = True,
            show_plot = False,
            preloaded_model = mdl
        )
        
        prediction = res_df.iloc[0,0]       
        
        return render_template('main.html',
                original_input={'CLOSE_VIX': CLOSE_VIX,
                                'AAII_SENT_BULLBEARSPREAD':AAII_SENT_BULLBEARSPREAD,
                                'YEARS_TO_NORMALIZATION':YEARS_TO_NORMALIZATION,
                                'IMPLIED_P_E': IMPLIED_P_E,
                },
                result=str(np.round(prediction,3)),
        )

@app.route('/doc',methods=['GET'])
def doc():
    if request.method == 'GET':
        return(render_template('doc.html')) # need to update this with every version update

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

if __name__ == '__main__':
    app.run()