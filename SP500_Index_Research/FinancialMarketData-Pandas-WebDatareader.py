# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 13:15:23 2019

https://pandas-datareader.readthedocs.io/en/latest/remote_data.html#fred

@author: SW
"""
# %% 1 setup

import pandas_datareader.data as web
import datetime

start = datetime.datetime(1776, 1, 1)
end = datetime.datetime(2019, 7, 27)

# %% 2 pull data FRED
# select security
# GDP, DGS10 , https://fred.stlouisfed.org/
security = 'DGS10'

data_pd = web.DataReader(security, 'fred',start, end)

#starts with Date
data_pd['Open'] = "" 
data_pd['High'] = "" 
data_pd['Low'] = "" 
data_pd = data_pd.rename(columns = {security : 'Close'})
data_pd['Volume'] = "" 
data_pd['Dividends'] = "" 
data_pd['Stock Splits'] = "" 
data_pd['Source'] = "" 
data_pd['LastUpdated'] = "" 
data_pd['fullExchangeName'] = "" 
data_pd['shortName'] = "" 
data_pd['currency'] = "" 
data_pd['type'] = "" 


data_pd['Security'] = security
data_pd['shortName'] = '10-Year Treasury Constant Maturity Rate' #UPDATE
data_pd['Source'] = 'FRED'
now = datetime.datetime.now()
data_pd['LastUpdated'] = now

data_pd = data_pd[['Open', 'High', 'Low', 'Close', 'Volume' , 'Dividends', 
        'Stock Splits', 'Security', 
        'Source' , 'LastUpdated', 'fullExchangeName',	'shortName'	,
        'currency', 'type']]

data_pd.to_csv(security + "_Daily_" + now.strftime("%Y-%m-%d") + ".csv")