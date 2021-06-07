# %% [markdown]
# H2O learning

# %%
# imports 
import h2o
from h2o.estimators import *
from h2o.grid import *

# %%
# start h2o

h2o.init()
#  h2o server default http://localhost:54321

# %%
# Import the dataset 
loan_level = h2o.import_file(
    "https://s3.amazonaws.com/data.h2o.ai/DAI-Tutorials/loan_level_500k.csv"
)

# %%
# Explore data

loan_level.describe()

# %%
# Explore specific

loan_level['DELINQUENT'].table()

# %%
# split into 3 sets
train, valid, test = loan_level.split_frame([0.7, 0.15], seed=42)

# %%
# check distr
print("train:%d valid:%d test:%d" % (train.nrows, valid.nrows, test.nrows))

# %%
# TODO data exploration + feature engineering

# %%
# choose vars 

y = 'DELINQUENT'
ignore = [
    'DELINQUENT', 'PREPAID', 'PREPAYMENT_PENALTY_MORTGAGE_FLAG', 'PRODUCT_TYPE',
    'LOAN_SEQUENCE_NUMBER',
]

x = list(set(train.names) - set(ignore))
print(x)

# %%
# build & train model

glm = H2OGeneralizedLinearEstimator(family ='binomial', seed = 42, model_id = 'default_glm')

%time glm.train(x = x, y = y, training_frame= train , validation_frame=valid)

# %%
# model summary

glm.summary()

# %%
# model training metrics

glm.training_model_metrics()

# %%
# plot scoring history

glm.plot(metric='negative_log_likelihood')

# %%
# variable importance

glm.varimp_plot()

# %%
# metrics accuracy

glm.accuracy()

# %%
# metric accuracy for specific threshold

glm.accuracy(thresholds = 0.9638505373028652)

# %%
# predict valid, using threshold at max_f1 score

glm.predict(valid).head(10)

# %%
# model perf valid 
default_glm_perf=glm.model_performance(valid)
# %%
# result
print(default_glm_perf.auc())

# %%
# TODO random forest


# %%
# Try GBM

gbm = H2OGradientBoostingEstimator(
    seed = 42, 
    model_id = 'default_gbm'
)

%time gbm.train(x = x, y = y, training_frame = train, validation_frame = valid)

# %%
# predict valid
gbm.predict(valid)

# %%
# GBM metrics
default_gbm_per = gbm.model_performance(valid)
print(default_gbm_per)

# %%
# # Try Tuned GLM

# glm_grid = h2o.grid.H2OGridSearch (
#     H2OGeneralizedLinearEstimator(family = "binomial",
#                                   lambda_search = True),
#     hyper_params = {"alpha": [x*0.01 for x in range(0, 100)],
#                     "missing_values_handling" : ["Skip", "MeanImputation"]},
#     grid_id = "glm_random_grid",
#     search_criteria = {
#         "strategy":"RandomDiscrete",
#         "max_models":300,
#         "max_runtime_secs":300,
#         "seed":42}
# )

# %time glm_grid.train(x = x, y = y, training_frame = train, validation_frame = valid)

# # %%
# # see results

# sorted_glm_grid = glm_grid.get_grid(sort_by = 'auc', decreasing = True)
# sorted_glm_grid.sorted_metric_table()

# # %%
# # see best module
# tuned_glm = sorted_glm_grid.models[0]
# tuned_glm.summary()

# # %%
# # eval model perf

# tuned_glm_perf = tuned_glm.model_performance(valid)

# print("Default GLM AUC: %.4f \nTuned GLM AUC:%.4f" % (default_glm_perf.auc(), tuned_glm_perf.auc()))


# # %%
# # See tuned results
# print ("Default GLM F1 Score:", default_glm_perf.F1())
# print ("Tuned GLM F1 Score", tuned_glm_perf.F1())
# print ("Default GLM: ", default_glm_perf.confusion_matrix())
# print ("Tuned GLM: ",  tuned_glm_perf.confusion_matrix())

# %%
# save GLM & reupload model

model_path = h2o.save_model(glm,path='../mlruns_h2o/',force=True)

print(model_path)

# load the model from server (if necesary)
# saved_model = h2o.load_model(model_path)

# download the model built above to your local machine (if necessary)
# my_local_model = h2o.download_model(saved_model, path="/Users/UserName/Desktop")

# upload the model that you just downloded above
# to the H2O cluster
uploaded_model = h2o.upload_model(model_path)

# %%
# Explain a model
exm = uploaded_model.explain(test)


# %%
# PDP 

pdp_table = uploaded_model.partial_plot(test,cols=['CREDIT_SCORE'], nbins = 20, plot=False)

# %%
# shutdown h2o server
h2o.cluster().shutdown(prompt=False) 

# %%
