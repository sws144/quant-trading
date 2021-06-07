# %% [markdown]
# ## G: Explain Models

# %%
# imports

import pandas as pd
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

import shap

import pickle

# %% 
# ### INPUT ###

runid = '870dc41593e7459da68839f3bebb2b86'

mlflow.set_experiment("P1-AnalyzeTrades_f_core")

# %%
# pull information

XY_df = pd.read_csv('output/e_resultcleaned.csv')
XY_df['weight'] = 1

# %%
# pull model from local

# TODO use actual function, otherwise need to update link
mdl = pickle.load(open(f'mlruns/0/{runid}/artifacts/model/model.pkl','rb'))

# %%
# pull information from mlflow

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
# parse data 

mlflow.start_run(run_id = runid )

y , X = dmatrices(formula_clean, XY_df, return_type='dataframe')

# add columns if necessary, can only add, not remove extra cols
cols_required = list(pd.DataFrame(
    json.loads(json.loads(tags['mlflow.log-model.history'])[0]['signature']['inputs'])
)['name'])

add_cols = list(set(cols_required) - set(list(X.columns)))
X[add_cols] = 0

X = X[cols_required] # ensure X is in correct order and complete for model

mlflow.end_run()

# %%
# train test, repeat from earlier 

X_train, X_test, y_train, y_test, XY_train, XY_test = train_test_split(
    X, y, XY_df, test_size=0.33, random_state=42)



# %% 
# summarize overall results

# Create object that can calculate shap values
explainer = shap.TreeExplainer(reg)

# save expected value
ev = explainer.expected_value[0]

# calculate shap values. This is what we will plot.
# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
shap_values = explainer.shap_values(X_train)

# Make plot. Index of [1] is explained in text below.
# shap.summary_plot(shap_values, X_train)

f = plt.gcf()

# Make plot to save
shap.summary_plot(shap_values, X_train,show=False,)
plt.tight_layout()
plt.savefig('summary_plot.png',bbox_inches = "tight")
plt.show()

mlflow.log_metric('expected_val', ev)
mlflow.log_artifact('summary_plot.png')

# %%
# check highest 

top_trades = XY_df[XY_df['PCT_RET_FINAL']>1]
top_trades.head()


# %%
# check top variable(s)

# make plots

plot_vars = ["Q('CLOSE_^VIX')", "Q('AAII_SENT_BULLBEARSPREAD')"]
for var in plot_vars:
    shap.dependence_plot(var, shap_values, X_train)

# %%
# end mlflow

# %%
# backup functions

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