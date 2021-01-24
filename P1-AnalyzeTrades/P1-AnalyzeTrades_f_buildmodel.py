# %% [markdown]
#  ## E: Build & Tune Models

# %%
# imports

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
# ### INPUTS ###
retune = False #hyperparameter tuning

# %% 
# start logging

# one time run to create
# mlflow.create_experiement()

mlflow.set_experiment("P1-AnalyzeTrades_f_core")

mlflow.sklearn.autolog()
mlflow.start_run()

mlflow.set_tag('run_id', mlflow.active_run().info.run_id)

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

# TODO save model formula, use this and na transformer
mlflow.log_params({"formula":formula})

# %% 
# train test data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


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
    # https://www.kaggle.com/eikedehling/tune-and-compare-xgb-lightgbm-rf-with-hyperopt

    # def gini(solution, submission):  # actual, expected
    #     """expects 2 lists"""                                       
    #     df = sorted(zip(solution, submission),    
    #             key=lambda x: x[1], reverse=True) # still a list, sorted by y_pred
    #     random = [float(i+1)/float(len(df)) for i in range(len(df))] # uniform percentiles             
    #     totalPos = np.sum([x[0] for x in df]) # sum of actual results                                      
    #     cumPosFound = np.cumsum([x[0] for x in df]) # list of cumulative actual                               
    #     Lorentz = [float(x)/totalPos for x in cumPosFound] # curve                        
    #     Gini = [l - r for l, r in zip(Lorentz, random)] # slice of diff from Lorenz and random                          
    #     return np.sum(Gini)                                                         

    def gini(actual, pred):
        assert (len(actual) == len(pred))
        allv = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
        allv = allv[np.lexsort((allv[:, 2], -1 * allv[:, 1]))]
        totalLosses = allv[:, 0].sum()
        giniSum = allv[:, 0].cumsum().sum() / totalLosses

        giniSum -= (len(actual) + 1) / 2.
        return giniSum / len(actual)

    # can swap in
    def gini_xgb(predictions, truth):
        truth = truth.get_label()
        return 'gini', -1.0 * gini(truth, predictions) / gini(truth, truth)

    # can swap in
    def gini_lgb(truth, predictions):
        score = gini(truth, predictions) / gini(truth, truth)
        return 'gini', score, True

    def gini_sklearn(truth, predictions):
        return gini(truth, predictions) / gini(truth, truth)

    gini_scorer = make_scorer(gini_sklearn, greater_is_better=True)

    # use hyperopt package with to better search 
    # https://github.com/hyperopt/hyperopt/wiki/FMin
    # use userdefined Gini, as it measures differentiation more
    def objective_gbr(params):
        "objective_gbr function for hyper opt, params is dict of params for mdl"
        parameters = {}
        for k in params:
            parameters[k] = int(params[k])
        mdl = GradientBoostingRegressor(random_state=0, **parameters)
        score = cross_val_score(mdl, X_train, y_train, scoring=gini_scorer, cv=5).mean()
        print("Gini {:.3f} params {}".format(score, parameters))
        return score

    # need to match estimator
    space = {
        'n_estimators': hp.quniform('n_estimators', 10, 100, 5),  # low # high # number of choices
        'max_depth': hp.quniform('max_depth', 2, 10, 2) 
    }

    best_params = fmin(fn=objective_gbr,
                space=space,
                algo=tpe.suggest,
                max_evals=5)

    print("Hyperopt estimated optimum {}".format(best_params))
else:
    best_params = {
        'n_estimators': 500, 
        'max_depth': 10
    }

# %%
# fit model 

reg = GradientBoostingRegressor(random_state=0, **best_params)
reg.fit(X_train,y_train)

# %%
# validate model

test_score_r2 = reg.score(X_test, y_test) # for GBM is r2

mlflow.log_metric("test_r2", test_score_r2)
print(test_score_r2)

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
shap.summary_plot(shap_values, X_train)

f = plt.gcf()

# Make plot. Index of [1] is explained in text below.
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

# make plot
shap.dependence_plot("Q('CLOSE_^VIX')", shap_values, X_train)

# %%
# end mlflow

mlflow.end_run()

# %%
# placeholder