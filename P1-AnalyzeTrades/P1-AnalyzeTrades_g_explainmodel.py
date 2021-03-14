# %% [markdown]
# ## G: Explain Models

# %%
# imports

import pandas as pd
import numpy as np
import mlflow
from patsy import dmatrices # for formula parsing

import json # for reading signature

from functools import partial

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor, GammaRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_tweedie_deviance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

from sklearn.metrics import mean_absolute_error, mean_squared_error, auc

import pickle

# %% 
# ### INPUT ###

runid = 'f71fb9cb2001496ba5cde6ce7a553bd3'

mlflow.set_experiment("P1-AnalyzeTrades_f_core")

# %%
# assist functions

# looks reasonable
# https://www.kaggle.com/jpopham91/gini-scoring-simple-and-efficient
def gini_normalized(y_true, y_pred, sample_weight=None):
    # check and get number of samples
    assert (np.array(y_true).shape == np.array(y_pred).shape, 
            'y_true and y_pred need to have same shape')
    n_samples = np.array(y_true).shape[0]
    
    # sort rows on prediction column 
    # (from largest to smallest)
    arr = np.array([y_true, y_pred, sample_weight]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]  # true col sorted by true
    pred_order = arr[arr[:,1].argsort()][::-1,0]  # true col sorted by pred
    
    true_order_wgts = arr[arr[:,0].argsort()][::-1,2] 
    pred_order_wgts = arr[arr[:,0].argsort()][::-1,2] 
    
    # get Lorenz curves
    L_true = (np.cumsum(np.multiply(true_order,true_order_wgts)) / 
        np.sum(np.dot(true_order,true_order_wgts)))
    L_pred = (np.cumsum(np.multiply(pred_order,pred_order_wgts)) / 
        np.sum(np.multiply(pred_order,pred_order_wgts)))
    L_ones = np.multiply(np.linspace(1/n_samples, 1, n_samples),pred_order_wgts)
    
    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    # normalize to true Gini coefficient
    return G_pred/G_true

def score_estimator(
    estimator, X_train, X_test, df_train, df_test, target, weights,
    tweedie_powers=None,):
    """
    Evaluate an estimator on train and test sets with different metrics
    """

    metrics = [
        ("default score", None),   # Use default scorer if it exists
        ("mean abs. error", mean_absolute_error),
        ("mean squared error", mean_squared_error),
        ("gini", gini_normalized)
    ]
    if tweedie_powers:
        metrics += [(
            "mean Tweedie dev p={:.4f}".format(power),
            partial(mean_tweedie_deviance, power=power)
        ) for power in tweedie_powers]

    res = []
    for subset_label, X, df in [
        ("train", X_train, df_train),
        ("test", X_test, df_test),
    ]:
        y, _weights = df[target], df[weights]
        for score_label, metric in metrics:
            if isinstance(estimator, tuple) and len(estimator) == 2:
                # Score the model consisting of the product of frequency and
                # severity models.
                est_freq, est_sev = estimator
                y_pred = est_freq.predict(X) * est_sev.predict(X)
            else:
                y_pred = estimator.predict(X)

            if metric is None:
                if not hasattr(estimator, "score"):
                    continue
                score = estimator.score(X, y, sample_weight=_weights)
            else:
                score = metric(y, y_pred, sample_weight=_weights)

            res.append(
                {"subset": subset_label, "metric": score_label, "score": score}
            )

    res = (
        pd.DataFrame(res)
        .set_index(["metric", "subset"])
        .score.unstack(-1)
        .round(4)
        .loc[:, ['train', 'test']]
    )
    return res

# %%
# pull information

XY_df = pd.read_csv('output/e_resultcleaned.csv')
XY_df['weight'] = 1

# %%
# pull model from local

mdl = pickle.load(open(f'mlruns/1/{runid}/artifacts/model/model.pkl','rb'))

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
# score estimator

score_estimator(mdl,X_train, X_test, XY_train, XY_test, formula_clean.split('~')[0].strip(),weights='weight')

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