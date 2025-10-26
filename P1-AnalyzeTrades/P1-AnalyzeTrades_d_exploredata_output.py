# %% [markdown]
# <span style="color:red; font-family:Helvetica Neue, Helvetica, Arial, sans-serif; font-size:2em;">An Exception was encountered at '<a href="#papermill-error-cell">In [10]</a>'.</span>

# %% [markdown]
#  # d explore data
#

# %% [markdown]
# ## Imports

# %%
#for formatting
import jupyter_black

jupyter_black.load(
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

# %% [markdown]
# <span id="papermill-error-cell" style="color:red; font-family:Helvetica Neue, Helvetica, Arial, sans-serif; font-size:2em;">Execution using papermill encountered an exception here and stopped:</span>

# %%
# TODO do different profiling
# from ydata_profiling import ProfileReport

# %%
profile = ProfileReport(df_datawattr, title="Report")
#profile #QA large size

# %%
