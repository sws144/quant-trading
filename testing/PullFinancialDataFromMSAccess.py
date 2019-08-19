"""
Pulls data from Access
https://github.com/mkleehammer/pyodbc/wiki/Connecting-to-Microsoft-Access
https://stackoverflow.com/questions/39835770/read-data-from-pyodbc-to-pandas
"""


import pyodbc
import pandas

# requires microsoft ODBC driver for use on Windows
# https://www.microsoft.com/en-US/download/details.aspx?id=13255
cnxn = pyodbc.connect(r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
                      r'DBQ=C:\Stuff\OneDrive\Data\FinancialMarketData.accdb;')
sql = "Select * " +  
      "From Daily"
data = pandas.read_sql(sql,cnxn)


# not working yet