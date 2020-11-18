# %% [markdown]
# ## Convert trade log into pnl design matrix for modeling
# design matrix is one record per row

# %%
# imports

import pandas as pd

# %%
# read in data
### INPUT ###
filename = 'PCM-Tracking - LogHist.csv'

df_raw = pd.read_csv(filename)

col_dict = {
    'DATE' : 'DATE',
    'CONTRACT' : 'CONTRACT',
    'TIME':'TIME',
    'PctReturn': 'PCTRETURN',
}


# %%
# clean data columns

df_cleaned = df_raw.copy(deep=True)
df_cleaned.columns = pd.Series(df_cleaned.columns.astype(str).str.upper().str.strip())
df_cleaned.columns = pd.Series(df_cleaned.columns).map(col_dict)\
    .fillna(pd.Series(df_cleaned.columns))

# QA
df_cleaned.columns 

# %%
# 