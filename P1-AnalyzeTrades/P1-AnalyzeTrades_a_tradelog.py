# %% [markdown]
#  ## Convert trade log into pnl design matrix for modeling
#  Design matrix is one record per row

# %%
# imports

import pandas as pd
import numpy as np # for np.nan
import glob # for text matching
import os # for path

import tradehelper as th # local class


# %%
# read in raw data
### INPUT ###
attr_filename = 'data/PCM-Tracking - LogHist.csv'

# trade + activity list, max 30 cols
globbed_files = glob.glob('data/*U106*.csv')
col_names_temp = list(range(30))
df_raw = pd.DataFrame(columns = col_names_temp)

# initial date
init_date = '2015-06-30'

for csv in globbed_files:
    frame = pd.read_csv(csv, names=col_names_temp)
    frame['filename'] = os.path.basename(csv)
    df_raw = df_raw.append(frame)

# attributes from trading
df_raw_attr = pd.read_csv(attr_filename)
df_raw_attr['filename'] = os.path.basename(attr_filename)
df_raw_attr = df_raw_attr.append(df_raw_attr)


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
# clean attribute columns

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
df_macro = df_clean_attr[~ df_clean_attr['ACTION'].astype(str).str.contains('BOT') & 
                      ~ df_clean_attr['ACTION'].astype(str).str.contains('SLD') &
                      ~ df_clean_attr['ACTION'].astype(str).str.contains('END')
                      ]
                    
df_clean_attr = df_clean_attr[ df_clean_attr['ACTION'].astype(str).str.contains('BOT') | 
                       df_clean_attr['ACTION'].astype(str).str.contains('SLD') |
                       df_clean_attr['ACTION'].astype(str).str.contains('END')
                        ]

df_clean_attr.head()


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
# update data types for attr

df_clean_attr['DATE'] = pd.to_datetime(df_clean_attr['DATE'],errors='coerce') 
numeric_cols = ['PRICE','COMMISSION','QTYCHG']
for col in numeric_cols:
    df_clean_attr[col] = (df_clean_attr[col].astype(str).str.strip()
        .str.replace('$','').str.replace(',','').astype(float)
        )

# QA
df_clean_attr.dtypes   

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
# save output

df_complete_trades.to_csv('output/a_completelog.csv')


# %%
