# %% [markdown]
#  ## E: Build & Tune Models

# %%
# imports

import pandas as pd
import numpy as np
from patsy import dmatrices

from sklearn.model_selection import train_test_split

# %%
# read in data

df_XY = pd.read_csv('output/e_resultcleaned.csv')

# %%
# Create formula for model

# formula = f'PCT_RET_FINAL ~ 1 + Q("IMPLIED_P/E")'
formula =  f""" PCT_RET_FINAL ~ 1 
    + IMPLIED_P_E + YEARS_TO_NORMALIZATION + Q('CLOSE_^VIX')
    + Q('%_TO_STOP') + Q('%_TO_TARGET') + Q('GROWTH_0.5TO0.75') + Q('ROIC_(BW_ROA_ROE)')

"""
        
y , X = dmatrices(formula, df_XY, return_type='dataframe')

# TODO save model formula, use this and na transformer


# %% 
# train test data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


# %%
# run model

from sklearn.ensemble import GradientBoostingRegressor

reg = GradientBoostingRegressor(random_state=0)
reg.fit(X_train, y_train)

# %%
# validate model

reg.score(X_test, y_test)


# %%
# visualize a single tree

from sklearn import tree

# Get a tree 
sub_tree_1 = reg.estimators_[1, 0]


tree.plot_tree(sub_tree_1,
           feature_names = list(X_train.columns),
           filled = True)

# %% 
# summarize overall results

import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(reg)

# calculate shap values. This is what we will plot.
# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
shap_values = explainer.shap_values(X_train)

# Make plot. Index of [1] is explained in text below.
shap.summary_plot(shap_values, X_train)

# %%
# check highest 

top_trades = df_XY[df_XY['PCT_RET_FINAL']>1]
top_trades.head()


# %%
# check top variable(s)

# make plot
shap.dependence_plot("Q('CLOSE_^VIX')", shap_values, X_train)

# %%
# scratch

# fig.savefig('imagename.png')

# from sklearn.tree import export_graphviz


# # Visualization. Install graphviz in your system
# # from pydotplus import graph_from_dot_data
# # from IPython.display import Image
# # dot_data = export_graphviz(
# #     sub_tree_1,
# #     out_file=None, filled=True, rounded=True,
# #     special_characters=True,
# #     proportion=False, impurity=False, # enable them if you want
# # )
# # graph = graph_from_dot_data(dot_data)
# # Image(graph.create_png())

# from dtreeviz.trees import *
# viz = dtreeviz(sub_tree_1,
#                X_train,
#                y_train.values,
#                target_name='PCT_RET_FINAL',
#                feature_names=list(X.columns))
              
# viz.view()    
# %%
