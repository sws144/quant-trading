# %% [markdown]
#  ## E: Build & Tune Models

# %%
# ### INPUTS ###
retune = True #hyperparameter tuning

# %%
# imports

import importlib  # for importing other packages
import pandas as pd
import numpy as np
from patsy import dmatrices

import mlflow # model tracking

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer

from hyperopt import hp, tpe
from hyperopt.fmin import fmin

# H2O , needs java installed & on path / environment variable (java --version to test)
import h2o
from h2o.estimators import *
from h2o.grid import *
from mlflow.models.signature import infer_signature

#evaluation 
from sklearn import tree
import shap  # package used to calculate Shap values
import matplotlib.pyplot as plt; 


# %% 
# start logging

# set location for mlruns
mlflow.set_tracking_uri('file:C:/Stuff/OneDrive/MLflow')

# set experiment
try:
    mlflow.set_experiment("P1-AnalyzeTrades_f_core") 
except:
    mlflow.create_experiment('P1-AnalyzeTrades_f_core')

mlflow.sklearn.autolog()

# %%
# read in data

df_XY = pd.read_csv('output/e_resultcleaned.csv')

# %%
# variables

target = 'PCT_RET_FINAL'
variables = [
    # 'UNNAMED:_0', 'UNNAMED:_0_X', 'UNNAMED:_0.1', 'QUANTITY', 'PNL',
    # 'OPEN_PRICE', 'CLOSE_PRICE', 'COMM_TOT', 'QTYCHG', 'PRICE',
    # 'COMMISSION', 'DETAILS', 'STOP', 'DAYSTOFYEND', 'FYEPSNXT',
    'IMPLIED_P_E', 
    'YEARS_TO_NORMALIZATION', 
    # 'CLOSE_^GSPC', 
    'CLOSE_^VIX',
    # 'UNNAMED:_0_Y', 'AAII_SENT_BULLISH', 'AAII_SENT_NEUTRAL',
    # 'AAII_SENT_BEARISH', 'AAII_SENT_TOTAL', 'AAII_SENT_BULLISH8WEEKMOVAVG',
    'AAII_SENT_BULLBEARSPREAD', 
    # 'AAII_SENT_BULLISHAVERAGE',
    # 'AAII_SENT_BULLISHAVERAGE+STDEV', 'AAII_SENT_BULLISHAVERAGESTDEV',
    # 'AAII_SENT_S&P500WEEKLYHIGH', 'AAII_SENT_S&P500WEEKLYLOW',
    # 'AAII_SENT_S&P500WEEKLYCLOSE', 
    '%_TO_STOP', '%_TO_TARGET',
    'GROWTH_0.5TO0.75', 'ROIC_(BW_ROA_ROE)', 
    # 'IMPLIED_P_E',
    # 'YEARS_TO_NORMALIZATION', 
    # 'OPEN_DATE', 'CLOSE_DATE', 'SYMBOL',
    # 'OPENACT', 'CLOSEACT', 'DATE', 'ACTION', 'TIME', 'UNNAMED:_6',
    # 'UNNAMED:_8', 'CASH_CHG_(PNL)', 'COMMENTS', 'PCTRETURN', 'STARTDATE',
    # 'COMPANY_NAME_(IN_ALPHABETICAL_ORDER)', 'TICKER', 'CURRENT_PRICE',
    # 'AT_PRICE', 'TARGET', 'EPS1', 'EPS2', 'FYEND', 'LASTUPDATED',
    # 'CATEGORY', 'COMMENTS.1', 'FILENAME', 'DATE_', 'AAII_SENT_DATE',
    # 'PCT_RET_FINAL'
]

dtypes =  pd.Series(df_XY.dtypes)
selected_vars = dtypes[variables]
varlist = []

cols_num = df_XY.select_dtypes(include='number').columns
idx = 0

for v in variables:
    if idx != 0:
        varlist.append('+')

    if v not in cols_num:
        varlist.append('C(')
    
    varlist.append("Q('")
    varlist.append(v)
    varlist.append("')")    
    
    if v not in cols_num:
        varlist.append(')')
    
    idx = idx + 1

varstring = ''.join(varlist)


# %%
# Create formula for model

formula =  f""" {target} ~ 1 + {varstring}"""
        
y , X = dmatrices(formula, df_XY, return_type='dataframe')

# %% 
# train test data

X_train, X_test, y_train, y_test, XY_train, XY_test = train_test_split(
    X, y.to_numpy().ravel(), df_XY, test_size=0.33, random_state=42)

# %%
# tune & run model using hyperopt


# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error
# from hpsklearn import HyperoptEstimator
# from hpsklearn import any_regressor
# from hpsklearn import any_preprocessing
# from hyperopt import tpe
# # load dataset
# url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
# dataframe = read_csv(url, header=None)
# # split into input and output elements
# data = dataframe.values
# data = data.astype('float32')
# X, y = data[:, :-1], data[:, -1]
# # split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# # define search
# model = HyperoptEstimator(regressor=any_regressor('reg'), preprocessing=any_preprocessing('pre'), loss_fn=mean_absolute_error, algo=tpe.suggest, max_evals=50, trial_timeout=30)
# # perform the search
# model.fit(X_train, y_train)
# # summarize performance
# mae = model.score(X_test, y_test)
# print("MAE: %.3f" % mae)
# # summarize the best model
# print(model.best_model())

    
# %%
# validate & log function

# looks reasonable
# https://www.kaggle.com/jpopham91/gini-scoring-simple-and-efficient
def gini_normalized(y_true, y_pred, sample_weight=None):
    # check and get number of samples
    assert np.array(y_true).shape == np.array(y_pred).shape, 'y_true and y_pred need to have same shape'
    n_samples = np.array(y_true).shape[0]
    
    # sort rows on prediction column 
    # (from largest to smallest)
    if sample_weight == None:
        sample_weight = np.ones(n_samples)
    
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
    estimator, X_train, X_test, df_train, df_test, target, formula, weights=None,
    tweedie_powers=None):
    """
    Evaluate an estimator on train and test sets with different metrics
    Requires active run on mlflow and estimator with .predict method
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, auc
    from functools import partial
    from sklearn.metrics import mean_tweedie_deviance
    
    mlflow.set_tag('run_id', mlflow.active_run().info.run_id)
    mlflow.log_params({"formula":formula})
    
    metrics = [
        # ("default score", None),   # Use default scorer if it exists
        ("mean abs. error", mean_absolute_error),
        ("mean squared error", mean_squared_error),
        ("gini", gini_normalized)
    ]
    if tweedie_powers:
        metrics += [(
            "mean Tweedie dev p={:.4f}".format(power),
            partial(mean_tweedie_deviance, power=power)
        ) for power in tweedie_powers]

    res = {}
    for subset_label, X, df in [
        ("train", X_train, df_train),
        ("test", X_test, df_test),
    ]:
        if weights != None:
            y, _weights = df[target], df[weights]
        else:
            y, _weights = df[target], None

        if isinstance(estimator, tuple) and len(estimator) == 2:
            # Score the model consisting of the product of frequency and
            # severity models.
            est_freq, est_sev = estimator
            y_pred = est_freq.predict(X) * est_sev.predict(X)
        elif 'h2o' in str(type(estimator)):
            y_pred = estimator.predict(h2o.H2OFrame(df)).as_data_frame().to_numpy().ravel() #ensure 1D array
        else:
            y_pred = estimator.predict(X)

        for score_label, metric in metrics:
            
            if metric is None:
                if not hasattr(estimator, "score"):
                    continue
                score = estimator.score(X, y, sample_weight=_weights)
            else:
                score = metric(y, y_pred, sample_weight=_weights)

            res[ score_label + '_'+ subset_label] = score 

    return res

# def log_w_validate(y_true, y_pred, formula:str = ''):
#     """validates reg model and log metrics to active mlflow run.
#     Requires active mlflow run to wrun

#     Args: \n
#         y_true (array): [actual results] \n
#         y_pred (array): [predicted results] \n
#     """
    
#     from sklearn import metrics
    
#     mlflow.set_tag('run_id', mlflow.active_run().info.run_id)
#     mlflow.log_params({"formula":formula})
    
#     test_score_r2 = metrics.r2_score(y_true, y_pred)

#     mlflow.log_metric("test_r2", test_score_r2)
#     print(f'r2 score {test_score_r2}')
    
#     return 

# %%
# fit model_1 boosting

mlflow.end_run()
mlflow.start_run(run_name='sklearn_gbm')

# tuning 
if retune:
                                                      
    func_f = importlib.import_module( "P1-AnalyzeTrades_f_buildmodel_func")

    gini_scorer = make_scorer(func_f.gini_sklearn, greater_is_better=True)

    # use hyperopt package with to better search 
    # https://github.com/hyperopt/hyperopt/wiki/FMin
    # use userdefined Gini, as it measures differentiation more
    def objective_gbr(params):
        "objective_gbr function for hyper opt, params is dict of params for mdl"
        mlflow.start_run(nested=True)
        parameters = {}
        for k in params:
            parameters[k] = int(params[k])
        mdl = GradientBoostingRegressor(random_state=0, **parameters)
        score = cross_val_score(mdl, X_train, y_train, scoring=gini_scorer, cv=5).mean()
        print("Gini {:.3f} params {}".format(score, parameters))
        mlflow.end_run()
        return score

    # need to match estimator
    space = {
        'n_estimators': hp.quniform('n_estimators', 10, 100, 5),  # low # high # number of choices
        'max_depth': hp.quniform('max_depth', 2, 4, 2) 
    }

    best_params = fmin(fn=objective_gbr,
                space=space,
                algo=tpe.suggest,
                max_evals=5)
    
    for key in best_params.keys():
        if int(best_params[key]) == best_params[key]:
           best_params[key] = int(best_params[key])

    print("Hyperopt estimated optimum {}".format(best_params))
        
else:
    best_params = {
        'n_estimators': 25, 
        'max_depth': 2
    }

reg = GradientBoostingRegressor(random_state=0, **best_params)
# reg = GradientBoostingRegressor(random_state=0)
reg.fit(X_train,y_train)

# log with validation
# log_w_validate(y_test, y_pred, formula)
res = score_estimator(reg,X_train, X_test, XY_train, XY_test, target, formula)

# addition artifacts 
# visualize a single tree
# Get a tree 
sub_tree_1 = reg.estimators_[0, 0]  # pull first 1 estimator, actual regressor vs array

tree.plot_tree(sub_tree_1,
           feature_names = list(X_train.columns),
           filled = True)

plt.tight_layout()
plt.savefig('tree_plot1.png',bbox_inches = "tight")
plt.show()

mlflow.log_artifact('tree_plot1.png')

mlflow.end_run()

# %%
# fit model_2 statsmodel TODO not working at moment

from sklearn.base import BaseEstimator, RegressorMixin
class SMWrapper(BaseEstimator, RegressorMixin):
    import numpy as np
    import statsmodels.api as sm

    """ A universal sklearn-style wrapper for statsmodels regressors """
    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()
    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return np.array(self.results_.predict(X)) # to ensure SHAP works

# from sklearn.datasets import make_regression
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LinearRegression

# X, y = make_regression(random_state=1, n_samples=300, noise=100)

# print(cross_val_score(SMWrapper(sm.OLS), X, y, scoring='r2'))

# %%
# Model 3 H2O GLM

h2o.init()
#  h2o server default http://localhost:54321

# full_df = h2o.import_file(
#     "output/e_resultcleaned.csv"
# )

glm = H2OGeneralizedLinearEstimator(family ='tweedie', seed = 42, model_id = 'H2O_model')
train = h2o.H2OFrame(XY_train) # use original full dataset
%time glm.train(x = variables, y = target, training_frame= train ) # column lists , then frame

glm.summary()

# TODO save model to mlflow with metrics

# test= h2o.H2OFrame(XY_test)

# default_glm_perf=glm.model_performance(test)

# QA
# complete = h2o.H2OFrame(df_XY)
# glm.explain(complete)

h2o.cluster().shutdown(prompt=False) 


# TODO SHAP not working

# import shap

# explainer = shap.SamplingExplainer(glm.predict,complete)
# ev = explainer.expected_value[0]

# def h2opredict(df): #df is pandas array with target
#     hf = h2o.H2OFrame(df) #pandas to h2o frame
#     predictions = glm.predict(test_data = hf) #predictions in h2o frame type
#     return predictions.as_data_frame().values

# df = complete.as_data_frame()
# explainer = shap.Explainer(h2opredict, df[list(glm.coef().keys())[1:]])
# shap_values = explainer(explainer(df[list(glm.coef().keys())[1:]][:100]))

# # explainer = shap.Explainer(reg)
# # shap_values = explainer.shap_values(df)
# shap.initjs()

# sample = 17 #explain 17th instance in the data set
 
# labels_pd = labels.as_data_frame()
# actual = labels_pd.iloc[sample].values[0]
# prediction = predictions_pd.iloc[sample]['predict']
# print("Prediction for ",sample,"th instance is ",prediction," whereas its actual value is ",actual)
 
# shap.force_plot(explainer.expected_value[prediction], shap_values[prediction][sample,:], df.iloc[sample])

# path <- "/path/to/model/directory"
# mojo_destination <- h2o.save_mojo(original_model, path = path)
# imported_model <- h2o.import_mojo(mojo_destination)

# new_observations <- h2o.importFile(path = 'new_observations.csv')
# h2o.predict(imported_model, new_observations)

# https://sefiks.com/2019/10/10/interpretable-machine-learning-with-h2o-and-shap/
# predictions = model.predict(test_data = hf)
# predictions.tail(5)
# predictions_pd = predictions['predict'].as_data_frame() #h2o frame to pandas

# %%
# Model 4 H2O GBM , tree-based

mlflow.end_run()
mlflow.start_run(run_name='H2O_gbm')

h2o.init()
#  h2o server default http://localhost:54321

# full_df = h2o.import_file(
#     "output/e_resultcleaned.csv"
# )

gbm = H2OGradientBoostingEstimator(seed = 42, model_id = 'H2O_model')
train = h2o.H2OFrame(XY_train) # use original full dataset
%time gbm.train(x = variables, y = target, training_frame= train ) # column lists , then frame


gbm.summary() 
# gbm._model_json['output']['names'] # coefs

test= h2o.H2OFrame(XY_test)

default_gbm_perf = gbm.model_performance(test)

res = score_estimator(gbm,X_train, X_test, XY_train, XY_test, target, formula)

mlflow.log_metrics(res)

# QA
# complete = h2o.H2OFrame(df_XY[variables+[target]])
# gbm.explain(complete)

path = "output/"
mojo_destination = gbm.save_mojo(path = path, force=True)
imported_model = h2o.import_mojo(mojo_destination)

mlflow.log_artifact(mojo_destination)

model_sig = infer_signature(X_train, y_train)

mlflow.h2o.log_model(gbm,path,signature=model_sig)

h2o.cluster().shutdown(prompt=False) 

mlflow.end_run()


# %%
# other validation functions

# https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_poisson_regression_non_normal_loss.html

# from sklearn.utils import gen_even_slices


# def _mean_frequency_by_risk_group(y_true, y_pred, sample_weight=None,
#                                   n_bins=10):
#     """Compare predictions and observations for bins ordered by y_pred.

#     We order the samples by ``y_pred`` and split it in bins.
#     In each bin the observed mean is compared with the predicted mean.

#     Parameters
#     ----------
#     y_true: array-like of shape (n_samples,)
#         Ground truth (correct) target values.
#     y_pred: array-like of shape (n_samples,)
#         Estimated target values.
#     sample_weight : array-like of shape (n_samples,)
#         Sample weights.
#     n_bins: int
#         Number of bins to use.

#     Returns
#     -------
#     bin_centers: ndarray of shape (n_bins,)
#         bin centers
#     y_true_bin: ndarray of shape (n_bins,)
#         average y_pred for each bin
#     y_pred_bin: ndarray of shape (n_bins,)
#         average y_pred for each bin
#     """
#     idx_sort = np.argsort(y_pred)
#     bin_centers = np.arange(0, 1, 1/n_bins) + 0.5/n_bins
#     y_pred_bin = np.zeros(n_bins)
#     y_true_bin = np.zeros(n_bins)

#     for n, sl in enumerate(gen_even_slices(len(y_true), n_bins)):
#         weights = sample_weight[idx_sort][sl]
#         y_pred_bin[n] = np.average(
#             y_pred[idx_sort][sl], weights=weights
#         )
#         y_true_bin[n] = np.average(
#             y_true[idx_sort][sl],
#             weights=weights
#         )
#     return bin_centers, y_true_bin, y_pred_bin


# print(f"Actual number of claims: {df_test['ClaimNb'].sum()}")
# fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
# plt.subplots_adjust(wspace=0.3)

# for axi, model in zip(ax.ravel(), [ridge_glm, poisson_glm, poisson_gbrt,
#                                    dummy]):
#     y_pred = model.predict(df_test)
#     y_true = df_test["Frequency"].values
#     exposure = df_test["Exposure"].values
#     q, y_true_seg, y_pred_seg = _mean_frequency_by_risk_group(
#         y_true, y_pred, sample_weight=exposure, n_bins=10)

#     # Name of the model after the estimator used in the last step of the
#     # pipeline.
#     print(f"Predicted number of claims by {model[-1]}: "
#           f"{np.sum(y_pred * exposure):.1f}")

#     axi.plot(q, y_pred_seg, marker='x', linestyle="--", label="predictions")
#     axi.plot(q, y_true_seg, marker='o', linestyle="--", label="observations")
#     axi.set_xlim(0, 1.0)
#     axi.set_ylim(0, 0.5)
#     axi.set(
#         title=model[-1],
#         xlabel='Fraction of samples sorted by y_pred',
#         ylabel='Mean Frequency (y_pred)'
#     )
#     axi.legend()
# plt.tight_layout()

# def lorenz_curve(y_true, y_pred, exposure):
#     y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
#     exposure = np.asarray(exposure)

#     # order samples by increasing predicted risk:
#     ranking = np.argsort(y_pred)
#     ranked_exposure = exposure[ranking]
#     ranked_pure_premium = y_true[ranking]
#     cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
#     cumulated_claim_amount /= cumulated_claim_amount[-1]
#     cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
#     return cumulated_samples, cumulated_claim_amount


# fig, ax = plt.subplots(figsize=(8, 8))

# y_pred_product = glm_freq.predict(X_test) * glm_sev.predict(X_test)
# y_pred_total = glm_pure_premium.predict(X_test)

# for label, y_pred in [("Frequency * Severity model", y_pred_product),
#                       ("Compound Poisson Gamma", y_pred_total)]:
#     ordered_samples, cum_claims = lorenz_curve(
#         df_test["PurePremium"], y_pred, df_test["Exposure"])
#     gini = 1 - 2 * auc(ordered_samples, cum_claims)
#     label += " (Gini index: {:.3f})".format(gini)
#     ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)

# # Oracle model: y_pred == y_test
# ordered_samples, cum_claims = lorenz_curve(
#     df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"])
# gini = 1 - 2 * auc(ordered_samples, cum_claims)
# label = "Oracle (Gini index: {:.3f})".format(gini)
# ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray",
#         label=label)

# # Random baseline
# ax.plot([0, 1], [0, 1], linestyle="--", color="black",
#         label="Random baseline")
# ax.set(
#     title="Lorenz Curves",
#     xlabel=('Fraction of policyholders\n'
#             '(ordered by model from safest to riskiest)'),
#     ylabel='Fraction of total claim amount'
# )
# ax.legend(loc="upper left")
# plt.plot()


# %%
# placeholder