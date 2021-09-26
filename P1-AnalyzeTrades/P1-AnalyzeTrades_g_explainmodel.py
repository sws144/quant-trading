# %% [markdown]
## ## G: Explain Models

# %%
## imports

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
import numpy as np
import mlflow
from patsy import dmatrices # for formula parsing

import json # for reading signature

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor, GammaRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

import h2o

import shap
from datetime import datetime 
import os 

import pickle
import dill

# %% 
# ### INPUT ###

# runid = '1b6b96ef3cb14b93b60af5f2a84eeb94' 
# mlflow.set_tracking_uri('')

# Research tracking
runid = '993aac766d164995bad3e1225fa78ef7'
mlflow.set_tracking_uri('file:C:/Stuff/OneDrive/MLflow')

experiment_name = 'P1-AnalyzeTrades_f_core'

mlflow.set_experiment(experiment_name)
experiment_details = mlflow.get_experiment_by_name(experiment_name)

# %%
## pull information

XY_df = pd.read_csv('output/e_resultcleaned.csv')
XY_df['weight'] = 1

# %%
## pull model from tracking uri

# tracking_uri = mlflow.get_tracking_uri()

artifact_loc = str(experiment_details.artifact_location).replace('file:','')\
        .replace('file:','')\
        .replace('///','')

# try pickle first, otherwise try H2O
try:
    mdl = pickle.load(open(f'{artifact_loc}/{runid}/artifacts/model/model.pkl','rb'))
except:
    # for h2o models
    h2o.init()
    mdl = h2o.import_mojo(f'{artifact_loc}/{runid}/artifacts/')
# 

# %%
## pull information from mlflow

# TODO H2O signature missing

mlflow.end_run()
mlflow.start_run(run_id = runid )

def parse_mlflow_info(run_info):
    metrics = run_info.data.metrics
    params = run_info.data.params
    tags = run_info.data.tags
    return metrics, params, tags

metrics , params, tags = parse_mlflow_info(mlflow.get_run(runid))

mlflow.end_run()

formula_clean = params['formula'].replace('\n','')

# %%
## parse data 

mlflow.end_run()
mlflow.start_run(run_id = runid )

y , X = dmatrices(formula_clean, XY_df, return_type='dataframe')

# add columns if necessary, can only add, not remove extra cols
cols_required = list(pd.DataFrame(
    json.loads(json.loads(tags['mlflow.log-model.history'])[0]['signature']['inputs'])
)['name'])

add_cols = list(set(cols_required) - set(list(X.columns)))
X[add_cols] = 0

# extra columns in dataset
print('extra columns in expanded dataset: '+  str(list(set(list(X.columns)) - set(cols_required))))

X = X[cols_required] # ensure X is in correct order and complete for model

mlflow.end_run()

# %%
## train test, repeat from earlier 

# X_train, X_test, y_train, y_test, XY_train, XY_test = train_test_split(
#     X, y, XY_df, test_size=0.33, random_state=42)


# %%
## predict full dataset

if 'h2o' in str(type(mdl)) :
    X_hf = h2o.H2OFrame(X)
    y_pred = mdl.predict(X_hf)
else:
    y_pred = mdl.predict(X)

# h2o.cluster().shutdown(prompt=False)  # if want to end earlier


# %% 
## class 

class H2ORegWrapper:
    def __init__(self, h2o_model, feature_names):
        self.h2o_model = h2o_model
        self.feature_names = feature_names
    def predict(self, X):
            if isinstance(X, pd.Series):
                X = X.values.reshape(1,-1)
            # self.dataframe= pd.DataFrame(X, columns=self.feature_names)
            self.predictions = self.h2o_model.predict(X).as_data_frame().values
            return self.predictions.astype('float64')

# %%
## lightgbm pipeline

def sub_gbm(X_hf, y_pred):
    """creates explainer based on H2O X & y frames
    """
    import lightgbm as lgb
    import shap
    import h2o

    X2_df = X_hf.as_data_frame()
    y2_pred_df = y_pred.as_data_frame()
    
    # TODO create categorical pipeline

    gbm_mdl = lgb.LGBMRegressor(n_jobs=-1)
    gbm_mdl.fit(X2_df, y2_pred_df)

    explainer = shap.Explainer(gbm_mdl.booster_)

    return explainer

# %% 
## summarize overall results

mlflow.end_run()
mlflow.start_run(run_id = runid )

# Create object that can calculate shap values
if 'H2O' in str(type(mdl)):
    # print('h2o explanation')
    # h2o_wrapper = H2ORegWrapper(mdl,X.columns)
    # explainer = shap.SamplingExplainer(h2o_wrapper.predict,h2o.H2OFrame(X[0:100]))

    # TODO use lightgbm with astype('category') to create train model
    
    explainer = sub_gbm(X_hf, y_pred)

else:
    # assume sklearn etc.
    explainer = shap.Explainer(mdl)

# save expected value
ev = explainer.expected_value[0]

# calculate shap values. This is what we will plot.
# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
# TODO get rid of sample below
shap_values = explainer(X) # gets full shap_value descriptions 

# Make plot. Index of [1] is explained in text below.
# shap.summary_plot(shap_values, X_train)

f = plt.gcf()

# Make plot to save
# shap.summary_plot(shap_values, X,show=False,) # not as informative as beeswarm

# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d%m%Y_%H%M%S")
print("date and time =", dt_string)	

shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
plt.savefig(f'summary_plot_{dt_string}.png',bbox_inches = "tight")
plt.show()

mlflow.log_metric('expected_val', ev)
mlflow.log_artifact(f'summary_plot_{dt_string}.png')

os.remove(f'summary_plot_{dt_string}.png')

mlflow.end_run()


# %%
## save explainer

mlflow.end_run()
mlflow.start_run(run_id = runid )

with open(f'explainer.pkl', 'wb') as handle:
    dill.dump(explainer, handle, recurse=True)

mlflow.log_artifact(f'explainer.pkl')

os.remove(f'explainer.pkl')

mlflow.end_run()

# %%
## check highest 

top_trades = XY_df[XY_df['PCT_RET_FINAL']>1]
top_trades.head()


# %%
## plot partial dependence for vars

# make plots

mlflow.end_run()
mlflow.start_run(run_id = runid )

# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%Y%m%d_%H%M%S")
print("date and time =", dt_string)	

for var in cols_required:
    fig, ax = plt.subplots()
    
    shap.dependence_plot(var, shap_values.values, X, ax=ax, show=False)
    
    int_labels , col_bins = pd.cut(X[var],bins=10, retbins=True, labels=False)
    
    shap_col_df = pd.DataFrame({
    var : col_bins[int_labels],
    'expected_value_shap' : shap_values.values[:,X.columns.get_loc(var)]
    })
    
    shap_col_grp_df = shap_col_df.groupby(var).mean()
    
    shap_col_grp_df.plot(ax=ax)
    
    # TODO add
    # shap.plots.partial_dependence(var, mdl.predict, X,  model_expected_value=True)
    # f = plt.gcf()
    
    plt.tight_layout()
    plt.savefig(f'shappdp_{var}_{dt_string}.png',bbox_inches = "tight")
    plt.show()
    
    mlflow.log_artifact(f'shappdp_{var}_{dt_string}.png')
    
    os.remove(f'shappdp_{var}_{dt_string}.png')
mlflow.end_run()

# %%
## end mlflow and h2o

mlflow.end_run()
try:
    h2o.cluster().shutdown(prompt=False)  # if want to end earlier
except:
    pass



# %%
## backup functions

# https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
# def gini(actual, pred, sample_weight = None):
#     #ignores weights
#     assert (len(actual) == len(pred))
#     allv = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
#     allv = allv[np.lexsort((allv[:, 2], -1 * allv[:, 1]))]
#     totalLosses = allv[:, 0].sum()
#     giniSum = allv[:, 0].cumsum().sum() / totalLosses

#     giniSum -= (len(actual) + 1) / 2.
#     return giniSum / len(actual)

# def gini_normalized(truth, predictions, sample_weight=None):
#     return gini(truth, predictions) / gini(truth, truth)

# https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python
# def gini(x, sample_weight=None):
#     # The rest of the code requires numpy arrays.
#     x = np.asarray(x)
#     if sample_weight is not None:
#         w = np.asarray(sample_weight)
#         sorted_indices = np.argsort(x)
#         sorted_x = x[sorted_indices]
#         sorted_w = w[sorted_indices]
#         # Force float dtype to avoid overflows
#         cumw = np.cumsum(sorted_w, dtype=float)
#         cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
#         return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / 
#                 (cumxw[-1] * cumw[-1]))
#     else:
#         sorted_x = np.sort(x)
#         n = len(x)
#         cumx = np.cumsum(sorted_x, dtype=float)
#         # The above formula, with all weights equal to 1 simplifies to:
#         return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

# def gini_normalized(y_actual, y_pred, sample_weight=None):
#     """
#     Gini coefficient based on two lists and optional weight list, all of same shape
#     """
#     ans = (gini(y_pred , sample_weight=sample_weight)
#            / gini(y_actual , sample_weight=sample_weight)
#     )
#     return ans