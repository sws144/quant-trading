# %% [markdown]
#  ## d explore data
#

# %%
# imports

import pandas as pd

# %%
# read in data

df_datawattr = pd.read_csv('output/c_resulttradewattr.csv') 

# %%
# see data
df_datawattr.head()

# %%
# statistics

df_datawattr.describe(include='all')


# %%
# columns

df_datawattr.dtypes

# %%
# placeholder
