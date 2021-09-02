import pytest
import importlib  
import pandas as pd
import numpy as np
# this method allows for imports with hyphens
# also loads model one time
main_file = importlib.import_module("P1-AnalyzeTrades_h_predictresult") 

def test_predict_1_basic():
    inputs = pd.DataFrame({
        "Q('CLOSE_^VIX')": [20],
    })
    
    res_df = main_file.predict_return(
        mlflow_tracking_uri = '', 
        experiment_name =  'P1-AnalyzeTrades_f_core', 
        run_id =  '1b6b96ef3cb14b93b60af5f2a84eeb94', 
        inputs = inputs)
    
    assert type(res_df) == pd.DataFrame  , 'result not dataframe' 
    
    assert type(res_df['predicted_ret'][0]) == np.float64   , 'first cell of predicted_ret not 0' 
    
def test_explain_1_basic():
    inputs = pd.DataFrame({
        "Q('CLOSE_^VIX')": [40],
    })
    
    res_df, shap_obj, shap_df, f = main_file.predict_return(
        mlflow_tracking_uri = '', 
        experiment_name =  'P1-AnalyzeTrades_f_core', 
        run_id =  '1140c5a2c378445ba06b77647d969345', 
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
        mlflow_tracking_uri = '', 
        experiment_name =  'P1-AnalyzeTrades_f_core', 
        run_id =  '0928f3d8912f4a49b71eb62795cb9e0b', 
        inputs = inputs, 
        explain = True)
    
    assert type(shap_df) == pd.DataFrame  , 'result not dataframe' 
    
    assert shap_df.shape[1] > 1, 'missing shap values' 
    
    # should not be null response
    assert res_df.isna().sum().sum() == 0