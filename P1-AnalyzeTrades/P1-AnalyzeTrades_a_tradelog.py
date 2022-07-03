# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: p1analyzetrades
#     language: python
#     name: p1analyzetrades
# ---

# %% [markdown]
# # Convert trade log
# 1. into basic pnl design matrix for modeling
# 1. Design matrix is one record per row
#

# %% [markdown]
# ## imports

# %%
import pandas as pd
import numpy as np  # for np.nan
import glob  # for text matching
import os  # for path

import tradehelper as th  # local class

pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 150)

# %% [markdown]
# ## read in raw data

# %%
# activity file csv export from Interactive Brokers, max 30 cols
globbed_files = glob.glob("data/*U106*.csv")
col_names_temp = list(range(30))
df_raw = pd.DataFrame(columns=col_names_temp)

# initial date
init_date = "2015-06-30"

for csv in globbed_files:
    frame = pd.read_csv(csv, names=col_names_temp)
    frame["filename"] = os.path.basename(csv)
    df_raw = df_raw.append(frame)


# %%
# Understand df_raw
df_raw.head()


# %%
# See df_raw available data
df_raw[0].value_counts()[:10]


# %% [markdown]
# ## Create trading list, after first activity file

# %%
df_trades = df_raw[df_raw[0] == "Trades"]
df_trades.columns = df_trades.iloc[0, :]
df_trades.columns = [*df_trades.columns[:-1], "filename"]
cols = df_trades.columns[~df_trades.columns.isin([np.nan])]
df_trades = df_trades[cols]
df_trades = df_trades[df_trades["Header"] == "Data"]
df_trades = df_trades[df_trades["filename"] != os.path.basename(globbed_files[0])]
df_trades.head()


# %% [markdown]
# ## create initial portfolio based on first activity file, add port to trades

# %%
df_port_init = df_raw[df_raw[0] == "Open Positions"]
df_port_init.columns = df_port_init.iloc[0, :]
df_port_init = df_port_init[df_port_init["Header"] == "Data"]
df_port_init.columns = [*df_port_init.columns[:-1], "filename"]
cols = df_port_init.columns[~df_port_init.columns.isin([np.nan])]
df_port_init = df_port_init[cols]

df_port_init = df_port_init[
    df_port_init["filename"] == os.path.basename(globbed_files[0])
]

df_port_init.head()

# add to trades
df_port_init["Date/Time"] = init_date
df_port_init["T. Price"] = df_port_init["Cost Price"]

df_trades = pd.concat([df_port_init, df_trades])


# %%
# update data types for trades & fill nas

df_trades["Date/Time"] = pd.to_datetime(df_trades["Date/Time"], errors="coerce")
numeric_cols = [
    "T. Price",
    "Comm/Fee",
    "Quantity",
]  # T. Price for opening trade includes comm
for col in numeric_cols:
    df_trades[col] = (
        df_trades[col]
        .astype(str)
        .str.strip()
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

df_trades["Comm/Fee"] = df_trades["Comm/Fee"].fillna(0)

# QA
df_trades.dtypes

# %%
# create trades action col and normalize quantity and add ratio for later
df_trades["Action"] = np.where(df_trades["Quantity"] > 0, "B", "S")
df_trades["Quantity"] = abs(df_trades["Quantity"])
df_trades["RatioNewOld"] = 1

# %% [markdown]
# ## consider corporate actions

# %%
# pull corp actions
df_corpact = df_raw[df_raw[0] == "Corporate Actions"]
df_corpact.columns = df_corpact.iloc[0, :]  # col name is at top of block
df_corpact = df_corpact[df_corpact["Header"] == "Data"]
df_corpact.columns = [*df_corpact.columns[:-1], "filename"]
cols = df_corpact.columns[~df_corpact.columns.isin([np.nan])]
df_corpact = df_corpact[cols]

df_corpact = df_corpact[~df_corpact["Description"].isna()]  # remove na's

# add cols to match trades
df_corpact["Symbol"] = (
    df_corpact["Description"]
    .str.split("(", expand=True)[0]
    .str.split(".", expand=True)[0]
)
df_corpact["Action"] = "CA"
df_corpact["Date/Time"] = pd.to_datetime(df_corpact["Date/Time"], errors="coerce")
condlist = [df_corpact["Description"].str.contains("Split"), True]
choicelist = ["Split", ""]
df_corpact["ActionType"] = np.select(condlist, choicelist)

df_splits = df_corpact.loc[df_corpact["ActionType"] == "Split", :]

df_splits["RatioNewOld"] = 1
df_splits.loc[:, "RatioNewOld"] = (
    df_splits["Description"]
    .astype(str)
    .str.split(" for ", expand=True)[0]
    .str.split(" FOR ", expand=True)[0]
    .str.split(" ", expand=True)
    .iloc[:, -1]
    .str[0:2]
    .astype(float)
) / (
    df_splits["Description"]
    .astype(str)
    .str.split(" for ", expand=True)
    .iloc[:, -1]
    .str.split(" FOR ", expand=True)[0]
    .str.split(" ", expand=True)[0]
    .str[0:2]
    .astype(float)
)

# sort by time
df_trades = pd.concat([df_trades, df_splits]).sort_values("Date/Time", ascending=True)

# ratio
df_trades["RatioNewOld"] = df_trades["RatioNewOld"].fillna(1)

# %% [markdown]
# ## create completed trade list

# %%
# QA
# df_trades = pd.read_csv('data-tests/tradelog2_corpact.csv')

tm = th.TradeManager(store_trades=True, print_trades=False)

tm.process_df(df_trades)

# list of trade objects
complete_trades = tm.get_copy_of_closed_trades()

# pushed to dataframe
df_complete_trades = pd.concat([x.to_df() for x in complete_trades]).reset_index(
    drop=True
)

tm.get_pnl()

# %%
# understand data types

df_complete_trades.dtypes

# %% [markdown]
# ## save output

# %%
df_complete_trades.to_csv("output/a_completelog.csv")


# %%
