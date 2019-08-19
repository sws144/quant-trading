# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 00:06:54 2019

FinancialMarketData from yahoo finance
https://aroussi.com/post/python-yahoo-finance

@author: SW
"""

# %% 1 required packages
import yfinance as yf
import pandas as pd
from datetime import datetime

# %% 2 pull data
security = yf.Ticker("msft")
print(security)

# get stock info
security.info

# example
security.info['sourceInterval']

# %% 3 edit data
data = security.history(period="max")

data_pd = pd.DataFrame(data)

data_pd['Security'] = security.info['symbol']
data_pd['Source'] = 'Yahoo'

now = datetime.now()
data_pd['LastUpdated'] = now

data_pd['fullExchangeName'] = security.info['fullExchangeName']
data_pd['shortName'] = security.info['shortName']
data_pd['currency'] = security.info['currency']
data_pd['type'] = security.info['quoteType']

# example
data_pd_head = data_pd.head()

# %% 4 export 
data_pd.to_csv(security.info['symbol'] + "_Daily_" + now.strftime("%Y-%m-%d") + ".csv")
