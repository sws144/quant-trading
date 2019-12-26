"""
Index research based on shiller dat

http://www.econ.yale.edu/~shisller/data.htm

Req'd inputs:
C:\Stuff\OneDrive\Data\FinancialMarketData.xlsx
"""

# %% 1 packages
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# for datetime plotting
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# more plotting
import seaborn as sns

# machine learning & numpy
import numpy as np
from sklearn import linear_model

# file export
import openpyxl

# %% 2 import & explore data
df = pd.read_excel('SP500_Index_Research_FinancialMarketData.xlsx', 
                   sheet_name = "YearlyMacro")

print(df.dtypes)

# plot data
# fix for using IPython/Spyder
# %matplotlib qt5
fig, ax  = plt.subplots()
ax.plot(df['DateFraction'],df['Price'])

# ax1 = ax.twinx()
# ax1.plot(summary['DGS10'], color = 'r')

# %% 3 create columns to complete analysis

# future returns (dependent variable)
df['SP500FwdYr01'] = df['Price'].shift(-12) #next year
df['SP500FwdYr01Returns']  = df['SP500FwdYr01'] /df['Price'] - 1

df['SP500FwdYr10'] = df['Price'].shift(-120) #next 10 yrs
df['SP500FwdYr10Returns']  = (df['SP500FwdYr10'] /df['Price'])**(.1) - 1

# pick independent variables
df['InflationTrailing5yrFactor'] = df['CPI'] / df['CPI'].shift(60) #check previous 

df['PERatio'] = df['Price'] / df['Earnings']
df['Earnings10yr'] = df['Earnings'].rolling(window = 120).mean()  # last 10 yrs
df['Earnings10yrGrowthRate'] = (df['Earnings']/df['Earnings'].shift(120))**(.1)-1 # last 10 yrs
df['Earnings10yrAdj'] = df['Earnings10yr'] * (1+df['Earnings10yrGrowthRate'])**5 \
    * df['InflationTrailing5yrFactor']


# such as target price 
df['Targetprice1yr']  = df['Earnings'] /(df['RateGS10']/100)
df['Targetprice1yrReturn'] = df['Targetprice1yr'] /df['Price']  -1 
df['Targetprice10yr']  = df['Earnings10yrAdj'] /(df['RateGS10']/100)
df['Targetprice10yrReturn'] = (df['Targetprice10yr'] /df['Price'])**(.1)  -1 

#trim file
df = df.dropna()

#check results
print(df.head())

# %% 4 plot data
#fig2, ax2  = plt.subplots()
#ax2.plot(df['PERatio'],df['SP500FwdYr01Returns']) #to fix
sns.pairplot(df[['DateFraction','RateGS10','PERatio']])
sns.pairplot(df[['DateFraction', 'RateGS10',
                 'SP500FwdYr10Returns','Earnings10yrGrowthRate',
                 'InflationTrailing5yrFactor']])

fig3, ax3  = plt.subplots()
sns.lineplot(df['DateFraction'],df['Targetprice10yrReturn'], color = 'b')
#ax3 = plt.twinx()
sns.lineplot(df['DateFraction'],df['SP500FwdYr10Returns'], color = 'g')
sns.lineplot(df['DateFraction'],df['InflationTrailing5yrFactor']**.2-1, color = 'r')
sns.lineplot(df['DateFraction'],df['RateGS10']/100, color = 'y')
fig3.legend(labels=['Targetprice10yrReturn','SP500FwdYr10Returns',
                    'InflationTrailing5yrFactorAsRate', 'RateGS10' ])

# export data
df.to_excel('SP500_Index_Research_Results.xlsx', sheet_name = 'sheet1',)

# check relationship with 

# %% 5 fit model

X_train = df.loc[:,['PERatio',
             'InflationTrailing5yrFactor', 'RateGS10']]

Y_train = df.loc[:,'SP500FwdYr10Returns']

clf = linear_model.RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X_train, Y_train)

params = clf.get_params(deep=True)

print("default (R^2) score:" + np.array2string(clf.score(X_train,Y_train)))
print("intercept: " + np.array2string(clf.intercept_))
print("coefficients: " + np.array2string(clf.coef_))

## TODO continuing fitting

# consider adding corporate bonds?