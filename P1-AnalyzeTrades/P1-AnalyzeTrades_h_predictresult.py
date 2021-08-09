# function for predicting one result
import pandas as pd
import mlflow
import h2o
import pickle
import json

def parse_mlflow_info(run_info):
    metrics = run_info.data.metrics
    params = run_info.data.params
    tags = run_info.data.tags
    return metrics, params, tags

def predict_return(
    mlflow_tracking_uri: str, 
    experiment_name: str, 
    run_id: str , 
    inputs: pd.DataFrame) -> pd.DataFrame:
    """Predict the return of model in decimal form

    Args:
        mlflow_tracking_uri (str): where mlflow runs sit
        
        experiment_name (str): runs are organized into experimetns
        
        run_id (str): specific run
        
        inputs (pd.DataFrame): raw dataframe (can be multiple rows)

    Returns:
        pct_return: dataframe of results 
    """    
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    experiment_details = mlflow.get_experiment_by_name(experiment_name) 

    mlflow.end_run()
    mlflow.start_run(run_id = run_id )


    # pull model from tracking uri
    artifact_loc = str(experiment_details.artifact_location)\
        .replace('file:','')\
        .replace('///','')

    metrics , params, tags = parse_mlflow_info(mlflow.get_run(run_id))

    # try pickle first, otherwise try H2O
    # try absolute, then relative location
    
    if 'sklearn' in tags['estimator_class']:
        try:
            mdl = pickle.load(open(f'{artifact_loc}/{run_id}/artifacts/model/model.pkl','rb'))
        except: # for testing
            mdl = pickle.load(open(f'mlruns/0/{run_id}/artifacts/model/model.pkl','rb'))
    else:
        # for h2o models
        h2o.init()
        try:
            mdl = h2o.import_mojo(f'{artifact_loc}/{run_id}/artifacts/')
        except:
            mdl = h2o.import_mojo(f'mlruns/0/{run_id}/artifacts/')

    # add columns if necessary, can only add, not remove extra cols
    cols_required = list(pd.DataFrame(
        json.loads(json.loads(tags['mlflow.log-model.history'])[0]['signature']['inputs'])
    )['name'])

    add_cols = list(set(cols_required) - set(list(inputs.columns)))
    inputs_copy = inputs.copy()
    inputs_copy[add_cols] = 0

    # extra columns in dataset
    # print('extra columns in expanded dataset: '+  str(list(set(list(inputs_copy.columns)) - set(cols_required))))

    inputs_copy = inputs_copy[cols_required] # ensure X is in correct order and complete for model

    mlflow.end_run()

    #consider later
    # formula_clean = params['formula'].replace('\n','')

    pct_return = mdl.predict(inputs_copy)
    
    pct_return_df = pd.DataFrame(pct_return, columns=['predicted_ret'])

    return pct_return_df





