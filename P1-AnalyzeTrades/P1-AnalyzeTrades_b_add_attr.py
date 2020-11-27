# %% [markdown]
#  ## add attributes to trade log
#  Design matrix is one record per row

# %%
# imports

import pandas as pd
import numpy as np # for np.nan
import os # for path

# %%
# read in raw data
### INPUT ###

# formatted tradelog
trades_filename = 'output/a_completelog.csv'
df_complete_trades = pd.read_csv(trades_filename)

# attributes 1 from own log
attr_filename = 'data/PCM-Tracking - LogHist.csv'
df_raw_attr = pd.read_csv(attr_filename)
df_raw_attr['filename'] = os.path.basename(attr_filename)
df_raw_attr = df_raw_attr.append(df_raw_attr)

# attributes 2 
# TODO 


# %%
# ensure date time for open for complete trades
df_complete_trades['Open_Date'] = pd.to_datetime(df_complete_trades['Open_Date'], errors='coerce')

# %%
# check complete trades

df_complete_trades.dtypes


# %%
# clean attribute columns 1

col_dict_attr = {
    'DATE' : 'DATE',
    'CONTRACT' : 'CONTRACT',
    'TIME':'TIME',
    'ACTION':'ACTION',
    'PRICE':'PRICE',
    'QTYCHG':'QTYCHG',
    'COMMISSION':'COMMISSION',
    
    'PCTRETURN': 'PCTRETURN',
}

df_clean_attr = df_raw_attr.copy(deep=True)
df_clean_attr.columns = pd.Series(df_clean_attr.columns.astype(str).str.upper().str.strip())
df_clean_attr.columns = pd.Series(df_clean_attr.columns).map(col_dict_attr)    .fillna(pd.Series(df_clean_attr.columns))

df_clean_attr['ACTION'] = df_clean_attr['ACTION'].astype(str).str.strip()

# pull out macro / non trades
df_macro = df_clean_attr[
    ~ df_clean_attr['ACTION'].astype(str).str.contains('BOT') & 
    ~ df_clean_attr['ACTION'].astype(str).str.contains('SLD') &
    ~ df_clean_attr['ACTION'].astype(str).str.contains('END')
]
                    
df_clean_attr = df_clean_attr[ 
    df_clean_attr['ACTION'].astype(str).str.contains('BOT') | 
    df_clean_attr['ACTION'].astype(str).str.contains('SLD') |
    df_clean_attr['ACTION'].astype(str).str.contains('END')
]

df_clean_attr.head()


# %%
# update data types for attr

df_clean_attr['DATE'] = pd.to_datetime(df_clean_attr['DATE'],errors='coerce') 
numeric_cols = ['PRICE','COMMISSION','QTYCHG']
for col in numeric_cols:
    df_clean_attr[col] = (
        df_clean_attr[col].astype(str).str.strip()
        .str.replace('$','').str.replace(',','').astype(float)
    )

# QA
df_clean_attr.dtypes   

# %% 
# merge attr to completed trades

df_comptrade_wattr = df_complete_trades.merge(
    df_clean_attr, how = 'left', left_on = ['Open_Date','Symbol'],
    right_on= ['DATE','CONTRACT'],suffixes=('','_a')       
)

# %%
# save output

df_comptrade_wattr.to_csv('output/b_completewattr.csv')


# %%
