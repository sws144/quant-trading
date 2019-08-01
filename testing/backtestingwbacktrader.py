# https://towardsdatascience.com/trading-strategy-back-testing-with-backtrader-6c173f29e37f

# for python 2 & 3 compatbility
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# code to auto add packages
import subprocess
import sys
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

# install relevant packages
install("backtrader")
install("backtrader[plotting]")

# start 
import datetime
import os.path
import backtrader as bt

class TestStrategy(bt.Strategy):
    d

# class MyStrategy(bt.Strategy): 
#     def __init__(self):
#         self.sma = bt.indicators.SimpleMovingAverage(period=15)

#     def next(self):
#         if self.sma > self.data.close:
#             #do something
#             pass #placeholder so that code runs
#         elif self.sma < self.data.close:
#             #do something else
#             pass
        
cerebro.broker.setcommission(commission = 0.001)