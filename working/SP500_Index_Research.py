"""
http://www.econ.yale.edu/~shisller/data.htm

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
df['SP500FwdYr01'] = df['Price'].shift(-12)
df['SP500FwdYr01Returns']  = df['SP500FwdYr01'] /df['Price'] - 1

df['SP500FwdYr10'] = df['Price'].shift(-120)
df['SP500FwdYr10Returns']  = (df['SP500FwdYr10'] /df['Price'])**(.1) - 1

# pick independent variables
df['PERatio'] = df['Price'] / df['Earnings']
df['Earnings10yr'] = df['Earnings'].rolling(window = 120).mean() 
df['Earnings10yrGrowthRate'] = .03 #(df['Earnings']/df['Earnings'].shift(-120))**(.1)-1
df['Earnings10yrAdj'] = df['Earnings10yr'] * (1+df['Earnings10yrGrowthRate'])**5
## TODO DO INFLATION ADJUSTED

# such as target price 
df['Targetprice1yr']  = df['Earnings'] /(df['RateGS10']/100)
df['Targetprice1yrReturn'] = df['Targetprice1yr'] /df['Price']  -1 
df['Targetprice10yr']  = df['Earnings10yrAdj'] /(df['RateGS10']/100)
df['Targetprice10yrReturn'] = df['Targetprice10yr'] /df['Price']  -1 


#check results
print(df.head())

# %% 4 plot data
fig2, ax2  = plt.subplots()
ax2.plot(df['PERatio'],df['SP500FwdYr01Returns']) #to fix
sns.pairplot(df[['DateFraction','RateGS10','PERatio']])
sns.pairplot(df[['DateFraction', 'RateGS10', 'Targetprice10yrReturn','SP500FwdYr01Returns',
                 'SP500FwdYr10Returns',]])

# check relationship with 


# todo p/e 

# consider adding corporate bonds?