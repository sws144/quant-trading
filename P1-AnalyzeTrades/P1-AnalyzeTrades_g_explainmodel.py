# %% [markdown]
# ## G: Explain Models

# %%
# imports

import pandas as pd
import numpy as np
import mlflow

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

# %% 
# ### INPUT ###

runid = 'f71fb9cb2001496ba5cde6ce7a553bd3'

mlflow.set_experiment("P1-AnalyzeTrades_f_core")

mlflow.start_run(run_id = runid )


# %%
# assist functions
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html


def score_estimator(
    estimator, X_train, X_test, df_train, df_test, target, weights,
    tweedie_powers=None,
):
    """Evaluate an estimator on train and test sets with different metrics"""

    metrics = [
        ("D² explained", None),   # Use default scorer if it exists
        ("mean abs. error", mean_absolute_error),
        ("mean squared error", mean_squared_error),
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
# visualize a single tree

from sklearn import tree

# Get a tree 
sub_tree_1 = reg.estimators_[1, 0]

tree.plot_tree(sub_tree_1,
           feature_names = list(X_train.columns),
           filled = True)

# to verify, going down left side means split is true


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

import matplotlib.pyplot as plt; 
f = plt.gcf()

# Make plot. Index of [1] is explained in text below.
shap.summary_plot(shap_values, X_train,show=False,)
plt.tight_layout()
plt.savefig('summary_plot.png',bbox_inches = "tight")
plt.show()

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

