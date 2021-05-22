# %% [markdown]
#  ## Convert trade log into basic pnl design matrix for modeling
#  Design matrix is one record per row

# %%
# imports

import pandas as pd
import numpy as np # for np.nan
import glob # for text matching
import os # for path

import tradehelper as th # local class

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

# %%
# read in raw data
### INPUT ###

# activity file csv export from Interactive Brokers, max 30 cols
globbed_files = glob.glob('data/*U106*.csv') 
col_names_temp = list(range(30))
df_raw = pd.DataFrame(columns = col_names_temp)

# initial date
init_date = '2015-06-30'

for csv in globbed_files:
    frame = pd.read_csv(csv, names=col_names_temp)
    frame['filename'] = os.path.basename(csv)
    df_raw = df_raw.append(frame)



# %%
# Understand df_raw
df_raw.head()


# %%
# See df_raw available data
df_raw[0].value_counts()[:10]




# %%
# Create trading list, after first activity file
df_trades = df_raw[df_raw[0]=='Trades']
df_trades.columns  = df_trades.iloc[0,:]
df_trades.columns = [*df_trades.columns[:-1], 'filename']
cols = df_trades.columns[~df_trades.columns.isin([np.nan])]
df_trades = df_trades[cols]
df_trades = df_trades[df_trades['Header'] == 'Data']
df_trades = df_trades[df_trades['filename'] != os.path.basename(globbed_files[0])]
df_trades.head()


# %%
# create initial portfolio based on first activity file, add port to trades
df_port_init = df_raw[df_raw[0]=='Open Positions']
df_port_init.columns  = df_port_init.iloc[0,:]
df_port_init = df_port_init[df_port_init['Header'] == 'Data']
df_port_init.columns = [*df_port_init.columns[:-1], 'filename']
cols = df_port_init.columns[~df_port_init.columns.isin([np.nan])]
df_port_init = df_port_init[cols]

df_port_init = df_port_init[df_port_init['filename'] == os.path.basename(globbed_files[0])]

df_port_init.head()

# add to trades
df_port_init['Date/Time'] = '2015-06-30'
df_port_init['T. Price'] = df_port_init['Cost Price']

df_trades = pd.concat([df_port_init, df_trades])


# %%
# update data types for trades & fill nas

df_trades['Date/Time'] = pd.to_datetime(df_trades['Date/Time'],errors='coerce') 
numeric_cols = ['T. Price','Comm/Fee','Quantity']
for col in numeric_cols:
    df_trades[col] = (df_trades[col].astype(str).str.strip()
        .str.replace('$','').str.replace(',','').astype(float)
        )
    
df_trades['Comm/Fee'] = df_trades['Comm/Fee'].fillna(0) 
    
# QA
df_trades.dtypes  

# %%
# create trades action col and normalize quantity
df_trades['Action'] = np.where(df_trades['Quantity'] > 0, 'B', 'S')
df_trades['Quantity'] = abs(df_trades['Quantity'])


# %%
# create completed trade list

tm = th.TradeManager(store_trades=True, print_trades=False)

tm.process_df(df_trades)

# list of trade objects
complete_trades = tm.get_copy_of_closed_trades() 

# pushed to dataframe
df_complete_trades = pd.concat([x.to_df() for x in complete_trades]).reset_index(drop=True)

tm.get_pnl()

# %%
# understand data types

df_complete_trades.dtypes

# %%
# save output

df_complete_trades.to_csv('output/a_completelog.csv')


# %%
