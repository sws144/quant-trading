# %% [markdown]
# ## Convert trade log into pnl design matrix for modeling
# design matrix is one record per row

# %%
# imports

import pandas as pd
import glob

# %%
# read in data
### INPUT ###
attr_filename = 'data/PCM-Tracking - LogHist.csv'

# trade + activity list, max 30 cols
df_raw = pd.concat([pd.read_csv(x, names=list(range(30))) for x in glob.glob('data/*U106*.csv')])

#QA type of columns

df_trades = df_raw[df_raw[0]=='Trades']
df_trades.columns  = df_trades.iloc[0,:]

# attributes from trading
df_raw_attr = pd.read_csv(attr_filename)


# %%
# clean data columns names

# col_dict_attr = {
#     'DATE' : 'DATE',
#     'CONTRACT' : 'CONTRACT',
#     'TIME':'TIME',
#     'ACTION':'ACTION',
#     'PRICE':'PRICE',
#     'QTYCHG':'QTYCHG',
#     'COMMISSION':'COMMISSION',
    
#     'PctReturn': 'PCTRETURN',
# }


# df_clean_attr = df_raw_attr.copy(deep=True)
# df_clean_attr.columns = pd.Series(df_clean_attr.columns.astype(str).str.upper().str.strip())
# df_clean_attr.columns = pd.Series(df_clean_attr.columns).map(col_dict_attr)\
#     .fillna(pd.Series(df_clean_attr.columns))

# QA
# df_clean_attr.columns 

# df_clean_attr['ACTION'] = df_clean_attr['ACTION'].astype(str).str.strip()

# # pull out macro / non trades
# df_macro = df_clean_attr[~ df_clean_attr['ACTION'].astype(str).str.contains('BOT') & 
#                       ~ df_clean_attr['ACTION'].astype(str).str.contains('SLD') &
#                       ~ df_clean_attr['ACTION'].astype(str).str.contains('END')
#                       ]
                    
# df_clean_attr = df_clean_attr[ df_clean_attr['ACTION'].astype(str).str.contains('BOT') | 
#                        df_clean_attr['ACTION'].astype(str).str.contains('SLD') |
#                        df_clean_attr['ACTION'].astype(str).str.contains('END')
#                         ]

# %%
# update data types

# df_clean_attr['DATE'] = pd.to_datetime(df_clean_attr['DATE'],errors='coerce') 
# numeric_cols = ['PRICE','COMMISSION','QTYCHG']
# for col in numeric_cols:
#     df_clean_attr[col] = (df_clean_attr[col].astype(str).str.strip()
#         .str.replace('$','').str.replace(',','').astype(float)
#         )



# QA
# df_clean_attr.dtypes   

# %%
# calculate pnl 

# create final pnl df
# df_complete_trades = pd.DataFrame()

# create portfolio with key: contract name, 
# item is nested dictionary of qty, cost, date (last updated)
# df_portfolio = {}


# for i, row in df_clean_attr.iterrows():
#     if row['ACTION'].find('BOT'):        
#         if row['CONTRACT'] in df_portfolio:
#             # if positive position in portfolio, add to it
#             # if df_portfolio[row['CONTRACT']] > 0:
#             #     df_portfolio[row['CONTRACT']] = (row['PRICE'] * row['QTYCHG'] - row['COMMISSION']
#             #                                     + row['CONTRACT']
#             #                                     )
#             # if negative position
#             # else:
#                 # pass
#         # if not in portfolio, add to it
#         else: 
#             df_portfolio[row['CONTRACT']] = {
#                 'cost': row['PRICE'] * row['QTYCHG'] - row['COMMISSION']
#             }
#     elif row['ACTION'].find('SLD') | row['ACTION'].find('END'):
#         if row['CONTRACT'] in df_portfolio:
#             # df_portfolio[row['CONTRACT']] = (row['CONTRACT'] - (
#             #     row['PRICE'] * row['QTYCHG'] - row['COMMISSION'])
#             #                                  )
#         else: 
#             # ignore errors for now
#             pass
    
#     #QA
#     # print(str(i) +  f'\n' + str(row))
#     if i > 100:
#         break
    
    

# %%
