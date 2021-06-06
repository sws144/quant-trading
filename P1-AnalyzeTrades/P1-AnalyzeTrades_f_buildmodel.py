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
    'IMPLIED_P_E', 'YEARS_TO_NORMALIZATION', 
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
    'YEARS_TO_NORMALIZATION', 
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y.to_numpy().ravel(), test_size=0.33, random_state=42)

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
# tune & run model

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
        'n_estimators': 500, 
        'max_depth': 10
    }
    
# %%
# validate & log function

def log_w_validate(y_true, y_pred, formula:str = ''):
    """validates reg model and log metrics to active mlflow run.
    Requires active mlflow run to wrun

    Args: \n
        y_true (array): [actual results] \n
        y_pred (array): [predicted results] \n
    """
    
    from sklearn import metrics
    
    mlflow.set_tag('run_id', mlflow.active_run().info.run_id)
    mlflow.log_params({"formula":formula})
    
    test_score_r2 = metrics.r2_score(y_true, y_pred)

    mlflow.log_metric("test_r2", test_score_r2)
    print(f'r2 score {test_score_r2}')
    return 

# %%
# fit model_1 boosting

mlflow.end_run()
mlflow.start_run(run_name='sklearn_gbm')

reg = GradientBoostingRegressor(random_state=0, **best_params)
# reg = GradientBoostingRegressor(random_state=0)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)

log_w_validate(y_test, y_pred, formula)

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

top_trades = df_XY[df_XY['PCT_RET_FINAL']>1]
top_trades.head()


# %%
# check top variable(s)

# make plots

plot_vars = ["Q('CLOSE_^VIX')", "Q('AAII_SENT_BULLBEARSPREAD')"]
for var in plot_vars:
    shap.dependence_plot(var, shap_values, X_train)

# %%
# end mlflow

mlflow.end_run()

# %%
# placeholder