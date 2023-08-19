# %% [markdown]
#  # d explore data
#

# %% [markdown]
# ## Imports

# %%
#for formatting
import jupyter_black

jupyter_black.load(
    lab=False,
)

# %%
# imports

import pandas as pd

# %%
# read in data

df_datawattr = pd.read_csv("output/c_resulttradewattr.csv")

# %% [markdown]
# ## Basic info

# %%
# see data
df_datawattr.head()

# %%
# statistics

df_datawattr.describe(include="all")


# %%
for c in df_datawattr.columns:
    print(f"trying {c}")
    try:
        temp_df = pd.pivot_table(
            df_datawattr, index=["Open_Year"], values=c, aggfunc="mean"
        )
        temp_df.plot()
    except:
        print(f"{c} didnt work")


# %%
# columns

df_datawattr.dtypes

# %%
df_datawattr.columns

# %%
# placeholder

# %% [markdown]
# ## Try profile

# %%
from pandas_profiling import ProfileReport

# %%
profile = ProfileReport(df_datawattr, title="Report")
#profile #QA large size

# %%
