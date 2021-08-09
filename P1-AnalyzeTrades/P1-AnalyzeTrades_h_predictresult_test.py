import pytest
import importlib  
import pandas as pd
import numpy as np
# this method allows for imports with hyphens
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