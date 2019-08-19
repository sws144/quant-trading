# https://docs.python.org/3/library/sqlite3.html

# download sqlite3 files to folder, add to path ,
# to create a database, type: sqlite3 <name>

# required packages (run in command line)
# pip install "sqlite3"

import sqlite3

conn = sqlite3.connect('C:\Stuff\OneDrive\Data\FinancialMarketData.db')
c = conn.cursor()

stock = ('MSFT')

# regular datapull
c.execute('SELECT * FROM OneMinuteBars') #WHERE Security =?', (stock,))
results = c.fetchall()

# using pandas
# pip install "pandas"
import pandas as pd
results_df = pd.read_sql_query("SELECT * FROM OneMinuteBars ", conn)
results_df.head()

# %% add new data

c.execute('''CREATE TABLE stocks
             (date text, trans text, symbol text, qty real, price real)''')