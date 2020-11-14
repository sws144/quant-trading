


# %% [markdown]
# # Analyze Trades from PCM

# %%
#prepare packages and more 

#code completion, click tab after starting
# %%
from IPython import get_ipython
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

import numpy as np
import pandas as pd

# %% [markdown]
# ## Import Data
# Import the LogHist tab from PCM-Tracking from Google Drive 

# %%
df = pd.read_excel('PCM-Tracking.xlsx', sheet_name='LogHist')
df.head()


# %%
df.describe(include = 'all')


# %%
# see what columsn there are
print(df.info())


# %%
# work with active trades with return first 
df_trades = df[df['Action'].notnull() & df['PctReturn'].notnull()]
print(df_trades.shape)
df_trades.head()


# %%
# designate variables from trades
y_var = 'PctReturn'
X_var = ['% to Stop','% to Target', 'Growth*0.5to0.75']

#fill na with 0
df_trades = df_trades.fillna(0)

y = df_trades[y_var]
X = df_trades[X_var]

print('dependent variables')
print(y)
print("\nindepedent variables")
print(X)

# %% [markdown]
# ## Try GLM first

# %%
# glm packages
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# make train test split
X_train, X_test, y_train, y_test = train_test_split(     X, y, test_size=0.33, random_state=42)

# use ridge glm with builtin cross validation  
model_glm = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))

#fit model
model_glm.fit(X_train,y_train)


# %%
# see coef
print(model_glm.coef_)


# %%
# evaluate model using R2, 1.0 is best value
from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(model_glm, X, y, cv=5, scoring='r2')

for data in cv_score:
    print('{:9.3f}'.format(data))

# %% [markdown]
# Since best R2 score is 1.0 and 0.0 is starting point, this model is substantially worse than normal

# %%
# predict
y_pred = model_glm.predict(X_test)

# actual vs predicted plot
import matplotlib.pyplot as plt
f, (ax0) = plt.subplots(1, 1, sharey=True)
ax0.scatter(y_test, y_pred)
ax0.set(xlabel='y_test', ylabel='y_pred')
ax0.set_title("actual (x) vs predicted (y)")

# # plot residuals
# from yellowbrick.datasets import load_concrete
# from yellowbrick.regressor import ResidualsPlot

# visualizer = ResidualsPlot(model_glm)

# visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
# visualizer.score(X_test, y_test)  # Evaluate the model on the test data
# visualizer.show()                # Finalize and render the figure


# %%



