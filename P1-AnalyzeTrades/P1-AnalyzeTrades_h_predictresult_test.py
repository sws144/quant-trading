import pytest
import importlib  
import pandas as pd
# this method allows for imports with hyphens
main_file = importlib.import_module("P1-AnalyzeTrades_h_predictresult") 

def predict_1_basic_test():
    inputs = pd.DataFrame({
        "Q('CLOSE_^VIX')": [20],
    })
    
    assert type(main_file.predict_return(
        mlflow_tracking_uri = '', 
        experiment_name =  'P1-AnalyzeTrades_f_core', 
        run_id =  '1b6b96ef3cb14b93b60af5f2a84eeb94', 
        inputs = inputs)) == float  , 'prediction not working' 