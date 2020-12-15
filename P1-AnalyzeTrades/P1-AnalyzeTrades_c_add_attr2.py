# %% [markdown]
#  ## add attributes 2 to trade log
#  Design matrix is one record per row

# %%
# imports

import pandas as pd
import numpy as np # for np.nan
import os # for path

import yfinance as yf

# %%
# read in raw data
### INPUT ###

# formatted tradelog
trades_filename = 'output/b_completewattr.csv'
df_raw = pd.read_csv(trades_filename)

# %% 
# pull data from yahoo finance

reload_data = True

tickers = ['^VIX' , '^GSPC']
if reload_data: 
    df_data = yf.download(
        ' '.join(tickers), 
        start="2010-01-01", end="2020-12-01", 
        group_by='Tickers'
    )
    df_data_formatted = df_data.stack(level=0).reset_index().rename(columns={'level_1':'Ticker'})
    df_data_formatted.to_csv('output/c_mktdata.csv')
else:
    df_data_formatted = pd.read_csv('output/c_mktdata.csv')

df_data_formatted.head()

# %%
# pull data from Quandl 

# TODO AAII Sentiment



# %%
# pivoted

df_data_pivot = df_data_formatted.pivot(
    index=['Date'],columns=['Ticker'],values=['Close'],
).reset_index()
df_data_pivot.columns = ['_'.join(col).strip() for col in df_data_pivot.columns.values]
df_data_pivot['Date_'] = pd.to_datetime(df_data_pivot['Date_'],errors='coerce')
df_data_pivot.head()

# %% 
# merge

df_source = df_raw.copy(deep=True)

df_source['Open_Date'] = pd.to_datetime(df_source['Open_Date'],errors='coerce')


# need to sort
df_source = df_source.sort_values(['Open_Date']) 
df_data_pivot = df_data_pivot.sort_values(['Date_'])

df_result = pd.merge_asof(
    df_source, df_data_pivot,left_on=['Open_Date'],right_on=['Date_']
)


# %%
# save output

df_result.to_csv('output/c_resulttradewattr.csv')


# %%
