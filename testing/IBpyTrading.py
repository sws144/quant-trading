# live trading backtrader 
# https://medium.com/@danjrod/interactive-brokers-in-python-with-backtrader-23dea376b2fc


from __future__ import (absolute_import, division, print_function,
                        unicode_literals) 
import backtrader as bt 

# required packages
# code to auto add packages
import subprocess
import sys
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

# Requirement: open interactive brokers

class St(bt.Strategy):

    def logdata(self):
        txt = []
        txt.append('{}'.format(len(self)))
           
        txt.append('{}'.format(
            self.data.datetime.datetime(0).isoformat())
        )
        txt.append('{:.2f}'.format(self.data.open[0]))
        txt.append('{:.2f}'.format(self.data.high[0]))
        txt.append('{:.2f}'.format(self.data.low[0]))
        txt.append('{:.2f}'.format(self.data.close[0]))
        txt.append('{:.2f}'.format(self.data.volume[0]))
        print(','.join(txt))

    def next(self):
        self.logdata()
    
    data_live = False
    
    def notify_data(self, data, status, *args, **kwargs):
        print('*' * 5, 'DATA NOTIF:', data._getstatusname(status),
          *args)
        if status == data.LIVE:
            self.data_live = True


def run(args=None):
        cerebro = bt.Cerebro(stdstats=False)
        store = bt.stores.IBStore(port=7496)    
        data = store.getdata(dataname='EUR.USD-CASH-IDEALPRO',
                         timeframe=bt.TimeFrame.Days)    
        cerebro.resampledata(data, timeframe=bt.TimeFrame.Days,
                         compression=10)
        cerebro.addstrategy(St)    
        # cerebro.broker = store.getbroker() # to enable actual trading
        cerebro.run()

if __name__ == '__main__':
    run()