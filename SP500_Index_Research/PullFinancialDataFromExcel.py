# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 16:52:17 2019

@author: SW
"""

# %% 1 packages
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# for datetime
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# %% 2 import & explore data
df = pd.read_excel('C:\Stuff\OneDrive\Data\FinancialMarketData.xlsx', sheet_name = "Daily")

df.dtypes

# pivot by date
summary = df.pivot(index = 'Date', columns = 'Security', values = 'Close')

#inverted isnull to pick data without nulls
summary = summary[~summary.isnull().any(axis=1)] 

# plot data
# fix for using IPython/Spyder
# %matplotlib qt5
fig, ax  = plt.subplots()
ax.plot(summary.index, summary['SPY'])
ax1 = ax.twinx()
ax1.plot(summary['DGS10'], color = 'r')
