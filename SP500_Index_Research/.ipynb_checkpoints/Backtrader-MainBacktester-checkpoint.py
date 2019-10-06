# orig file https://www.backtrader.com/home/helloalgotrading/

# %% package
# required packages
# code to auto add packages
import subprocess
import sys
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

# %% req'd imports 
install("backtrader")
from datetime import datetime
import backtrader as bt
import backtrader.analyzers as btanalyzers

# Create a subclass of Strategy to define the indicators and logic
class SmaCross(bt.Strategy):
    # list of parameters which are configurable for the strategy
    params = dict(
        pfast=10,  # period for the fast moving average
        pslow=30   # period for the slow moving average
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        sma1 = bt.ind.SMA(period=self.p.pfast)  # fast moving average
        sma2 = bt.ind.SMA(period=self.p.pslow)  # slow moving average
        self.crossover = bt.ind.CrossOver(sma1, sma2)  # crossover signal
        
    def next(self):
        self.log('Close, %.2f' % self.dataclose[0])
        if not self.position:  # not in the market
            if self.crossover > 0:  # if fast crosses slow to the upside
                self.buy(size=100)  # enter long
                self.log('BUY CREATE, %.2f' % self.dataclose[0])

        elif self.crossover < 0:  # in the market & cross to the downside
            self.close()  # close long position

        if len(self.data) == self.data.buflen()-1:
            self.close()

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

# %% Run main
if __name__ == '__main__':
    cerebro = bt.Cerebro()  # create a "Cerebro" engine instance

    # Create a data feed
    data = bt.feeds.YahooFinanceData(dataname='MSFT',
                                    fromdate=datetime(2011, 1, 1),
                                    todate=datetime(2011, 12, 31))

    cerebro.adddata(data)  # Add the data feed
    cerebro.addstrategy(SmaCross)  # Add the trading strategy

    
    cerebro.broker.setcash(10000.0)
    print('Starting Portfolio Value: %.2f' %
            cerebro.broker.getvalue()
    )
    
    # Analyzer
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='mysharpe',timeframe=bt.TimeFrame.Days)

    # Run strategy    
    thestrats = cerebro.run()
    thestrat = thestrats[0]

    print('Final Portfolio Value: %.2f' %
        cerebro.broker.getvalue()
    )

    print('Sharpe Ratio: %.4f' % thestrat.analyzers.mysharpe.get_analysis()['sharperatio'])
    cerebro.plot()  # and plot it with a single command


