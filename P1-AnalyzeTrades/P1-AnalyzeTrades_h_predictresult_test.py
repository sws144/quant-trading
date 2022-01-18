import os
import pytest
import importlib  
import pandas as pd
import numpy as np
# this method allows for imports with hyphens
# also loads model one time
main_file = importlib.import_module("P1-AnalyzeTrades_h_predictresult") 

# current file
mlflow_tracking_uri =  '' # backup file:' + os.path.dirname(os.path.abspath(__file__))
# if os.path.isdir('D:/Stuff/OneDrive/MLflow'):
#     mlflow_tracking_uri = 'file:D:/Stuff/OneDrive/MLflow'

def test_predict_1_basic():
    inputs = pd.DataFrame({
        "Q('CLOSE_^VIX')": [20],
    })
    
    res_df = main_file.predict_return(
        mlflow_tracking_uri = mlflow_tracking_uri, 
        experiment_name =  'P1-AnalyzeTrades_f_core', 
        run_id =  '5011fe3fca374334a349d0eff25ab317', 
        inputs = inputs)
    
    assert type(res_df) == pd.DataFrame  , 'result not dataframe' 
    
    assert (type(res_df['predicted_ret'][0]) == np.float64   , 
            'first cell of predicted_ret not 0') 
    
def test_explain_1_basic():
    """sklearn test
    """
    inputs = pd.DataFrame({
        "Q('CLOSE_^VIX')": [40],
    })
    
    res_df, shap_obj, shap_df, f = main_file.predict_return(
        mlflow_tracking_uri = mlflow_tracking_uri, 
        experiment_name =  'P1-AnalyzeTrades_f_core', 
        run_id =  '5011fe3fca374334a349d0eff25ab317', 
        inputs = inputs, 
        explain = True)
    
    assert type(shap_df) == pd.DataFrame  , 'result not dataframe' 
    
    assert shap_df.shape[1] > 1, 'missing shap values'
    
    # should not be null response
    assert res_df.isna().sum().sum() == 0 
  
def test_explain_2_h2omodel():
    inputs = pd.DataFrame({
        "Q('CLOSE_^VIX')": [40],
    })
    
    res_df, shap_obj, shap_df, f = main_file.predict_return(
        mlflow_tracking_uri = mlflow_tracking_uri, 
        experiment_name =  'P1-AnalyzeTrades_f_core', 
        run_id =  'f1d3ba19eb3646aa81eb67e5a75dab43', 
        inputs = inputs, 
        explain = True)
    
    assert type(shap_df) == pd.DataFrame  , 'result not dataframe' 
    
    assert shap_df.shape[1] > 1, 'missing shap values' 
    
    # should not be null response
    assert res_df.isna().sum().sum() == 0