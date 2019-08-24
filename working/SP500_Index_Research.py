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
df['SP500FwdYr01Returns']  = df['SP500FwdYr01'] /df['Price'] 

# pick independent variables


#check results
print(df.head())

# todo p/e 

# consider adding corporate bonds?