# %% [markdown]
#  ## C add attributes 2 and begin feat engineering
#  Design matrix is one record per row
#

# %%
# formatting
import jupyter_black

jupyter_black.load(
    lab=False,
)

# %%
# imports

import pandas as pd
import numpy as np  # for np.nan
import os  # for path

import yfinance as yf

import json

from pandas_datareader.quandl import QuandlReader  # data side

reload_data = True
end_date = "2030-12-31"

# %%
with open("data/vars.json", "r") as json_file:
    var_dict = json.load(json_file)

# %%
# read in raw data
### INPUT ###

# formatted tradelog
trades_filename = "output/b_completewattr.csv"
df_raw = pd.read_csv(trades_filename)

# %% [markdown]
# ### Existing Cols with Issues

# %%
existing_err_cols = [
    "AAII_SENT_Date",
    "AAII_SENT_Bullish",  ## AAII columns are missing later 2021 and 2022 values
    "AAII_SENT_Neutral",
    "AAII_SENT_Bearish",
    "AAII_SENT_Total",
    "AAII_SENT_Bullish8WeekMovAvg",
    "AAII_SENT_BullBearSpread",
    "AAII_SENT_BullishAverage",
    "AAII_SENT_BullishAverage+StDev",
    "AAII_SENT_BullishAverageStDev",
    "AAII_SENT_S&P500WeeklyHigh",
    "AAII_SENT_S&P500WeeklyLow",
    "AAII_SENT_S&P500WeeklyClose",
]

# %% [markdown]
# ## pull data from yahoo finance

# %%
tickers = ["^VIX", "^GSPC"]
if reload_data:
    df_data = yf.download(
        " ".join(tickers), start="2010-01-01", end=end_date, group_by="Tickers"
    )
    # turn into tabular form
    df_data_formatted = (
        df_data.stack(level=0).reset_index().rename(columns={"level_1": "Ticker"})
    )
    df_data_formatted.to_csv("output/c_mktdata.csv")
else:
    df_data_formatted = pd.read_csv("output/c_mktdata.csv")

df_data_formatted.head()


# %%
# pivoted

df_data_pivot = df_data_formatted.pivot(
    index=["Date"],
    columns=["Ticker"],
    values=["Close"],
).reset_index()

# to deal with multiindex columns
df_data_pivot.columns = ["_".join(col).strip() for col in df_data_pivot.columns.values]
df_data_pivot.rename(columns={'Date_':'Date_YahooFinance'}, inplace=True)
df_data_pivot["Date_YahooFinance"] = pd.to_datetime(df_data_pivot["Date_YahooFinance"], errors="coerce")
df_data_pivot.head()

# %%
# merge

df_source = df_raw.copy(deep=True)

df_source["Open_Date"] = pd.to_datetime(df_source["Open_Date"], errors="coerce")


# need to sort
df_source = df_source.sort_values(["Open_Date"])
df_data_pivot = df_data_pivot.sort_values(["Date_YahooFinance"])

df_result = pd.merge_asof(
    df_source, df_data_pivot, left_on=["Open_Date"], right_on=["Date_YahooFinance"]
)


# %% [markdown]
# ### Add Open Year

# %%
df_result["Open_Year"] = df_result["Open_Date"].dt.year

# %% [markdown]
# ### Test data

# %%
df_result.columns

# %%
for symbol in tickers:
    temp_df = pd.pivot_table(
        df_result,
        index=["Open_Year"],
        values="Close_" + symbol,
        aggfunc="mean",
        dropna=False,
    )
    temp_df.plot()

    assert min(temp_df["Close_" + symbol]) > 0, "values should all be above zero"


# %% [markdown]
# ## pull data from Quandl  / Nasdaq Data Link
# 1. log in is from [Nasdaq data link](https://docs.data.nasdaq.com/docs/python-installation)

# %%
externalvar_dict = {
    "AAII/AAII_SENTIMENT": "AAII_SENT",  ## aaii sentiment
    #     "FED/RIMLPAAAR_N_B": "FED_AAACORP",  ## daily Fed AAA rates #TODO ned to fix different timeframes
}

# %%
# import nasdaqdatalink

# NASDAQ_DATA_LINK_API_KEY = var_dict["NASDAQ_DATA_LINK_API_KEY"]
# data = nasdaqdatalink.get("AAII/AAII_SENTIMENT", start_date="2015-01-01", end_date="2030-12-31")

# %%
externalvar_dict = {
    "AAII/AAII_SENTIMENT": "AAII_SENT",  ## aaii sentiment
    #     "FED/RIMLPAAAR_N_B": "FED_AAACORP",  ## daily Fed AAA rates #TODO ned to fix different timeframes
}

# %%
if reload_data:

    #     df_varlist = []

    for variable, value in externalvar_dict.items():

        quandl_key = var_dict["QUANDL_API"]

        QR = QuandlReader(variable, api_key=quandl_key, start="1/1/2015", end=end_date)

        QR_df = QR.read().reset_index()

        QR_df.columns = [
            value + "_" + str(col)  # if col.upper() != "DATE" else col
            for col in QR_df.columns
        ]

        #         df_varlist.append(QR_df)

        # merge Quandl
        QR_df_sorted = QR_df.sort_values([f"{value}_Date"])
        QR_df_sorted["Date"] = pd.to_datetime(QR_df[f"{value}_Date"], errors="coerce")

        df_result = pd.merge_asof(
            df_result,
            QR_df_sorted,
            left_on=["Open_Date"],
            right_on=[f"{value}_Date"],
            direction="forward",
        )

        #     QR_df = pd.concat(
        #         [df.set_index("Date") for df in df_varlist], axis=1, join="outer"
        #     ).reset_index()
        QR_df.to_csv(f"output/c_quandl_{value}.csv")
else:
    for variable, value in externalvar_dict.items():

        QR_df = pd.read_csv(f"output/c_quandl_{value}.csv")

        # merge Quandl
        QR_df_sorted = QR_df.sort_values(["Date"])
        QR_df_sorted["Date"] = pd.to_datetime(QR_df[f"{value}_Date"], errors="coerce")

        df_result = pd.merge_asof(
            df_result,
            QR_df_sorted,
            left_on=["Open_Date"],
            right_on=[f"{value}_Date"],
            direction="forward",
        )


# %%
QR_df.head()

# %%
cols_with_errors = []

for variable, value in externalvar_dict.items():
    for c in df_result.columns:
        if value in c:
            temp_df = pd.pivot_table(
                df_result, index=["Open_Year"], values=c, aggfunc="mean", dropna=False
            )
            temp_df.plot()

            try:
                assert sum(temp_df[c].isna()) == 0, "values should all be filled"
            except:
                cols_with_errors.append(c)


# %%
cols_with_errors

# %%
temp_df

# %%
# check diff
assert len(set(cols_with_errors).difference(set(existing_err_cols))) == 0, "new errors"

# %% [markdown]
# ## Final Checks

# %%
df_result.loc[0, df_result.columns.duplicated()]

# %%
assert len(df_result.loc[0, df_result.columns.duplicated()]) == 0, "duplicates"

# %% [markdown]
# ## Save Output

# %%
df_result.to_csv("output/c_resulttradewattr.csv")


# %%
