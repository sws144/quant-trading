# %% trading strategy backtesting with backtrader
# https://towardsdatascience.com/trading-strategy-back-testing-with-backtrader-6c173f29e37f

# req'd packages
# pip install "package"

from __future__ import (absolute_import, division, print_function, 
                        unicode_literals) #for python 2 compatibility

from datetime import datetime
import os.path
import sys
import backtrader as bt
#import matplotlib 


# %% define strategy
class TestStrategy(bt.Strategy): #class that inherits bt.strategy
    def __init__(self):      
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
 
        self.sma = bt.indicators.MovingAverageSimple(self.datas[0], period = 15)
        self.rsi = bt.indicators.RelativeStrengthIndex()
    
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' % 
                    (order.executed.price, order.executed.value, 
                     order.executed.comm)
                )
            else: #sell
                self.log(
                    'SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' % 
                    (order.executed.price, order.executed.value, 
                     order.executed.comm)
                )
            
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            
            #write down
            self.order = None
            
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        
        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm)
        )
        
    def next(self):
        self.log('Close, %.2f' % self.dataclose[0])
        print('rsi:', self.rsi[0])
        
        if self.order:
            
            return #do nothing if have outstanding order
        
        if not self.order:
            if (self.rsi[0] < 30):
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                self.order = self.buy(size = 1)
        else:
            if (self.rsi[0] > 70):
                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                self.order = self.sell(size = 1)
                    
        
# %% initialize backtester

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(TestStrategy)
    cerebro.broker.setcommission(commission=0.001)
    
    data = bt.feeds.YahooFinanceData(dataname='MSFT',
                                 fromdate=datetime(2011, 1, 1),
                                 todate=datetime(2012, 12, 31))
        
    cerebro.adddata(data)
    cerebro.broker.setcash(10000.0)
    print('Starting Portfolio Value: %.2f' %
          cerebro.broker.getvalue()
    )
    cerebro.run()
    print('Final Portfolio Value: %.2f' %
          cerebro.broker.getvalue()
    )
    cerebro.plot()

    

        