# leverage function from online

import csv
import sys
import pandas as pd
from collections import defaultdict, deque

# ASSUMPTIONS MADE
# only one trade for a particular symbol at one time tick
# csv always sorted by time (instructions kind of say this, unclear) 
#   necessary or else my usage of deque is not right
# can't trade for fractions of cent
# csv always right: price: 2 degree of precision float, quantity: int

class TradeManager():
    def __init__(self, store_trades=True, print_trades=False):
        self._open_trades = defaultdict(deque)
        self._closed_trades = []
        self._store_trades = store_trades
        self._print_trades = print_trades
        self._pnl = 0.0

    def process_trade(self, trade):
        d = self._open_trades[trade.symbol]

        # if no inventory, just add it
        if len(d) == 0:
            d.append(trade)
            return

        # if inventory exists, all trades must be same way (buy or sell)
        # if new trade is same way, again just add it
        if d[0].buying == trade.buying:
            d.append(trade)
            return

        # otherwise, consume the trades
        while len(d) > 0 and trade.quantity > 0:
            quant_traded = min(trade.quantity, d[0].quantity)
            
            pnl = quant_traded * round(trade.price - d[0].price, 2)
            # invert if we shorted
            if trade.buying:
                pnl *= -1

            #commission 
            pnl += trade.comm  # comm is negative number

            pnl = round(pnl, 2)
            self._pnl += pnl

            ct = ClosedTrade(d[0].time, trade.time, trade.symbol,
                    quant_traded, pnl, d[0].buying, d[0].price, trade.price,
                    d[0].comm + trade.comm
                    )

            if self._print_trades:
                print(ct)
            if self._store_trades:
                self._closed_trades.append(ct)

            trade.quantity -= quant_traded
            d[0].quantity -= quant_traded
            
            if d[0].quantity == 0:
                d.popleft()

        # if the new trade still has quantity left over
        # then add it
        if trade.quantity > 0:
            d.append(trade)

    # NEED TO UPDATE FOR SPECIFIC FILES
    def process_csv(self, file_name):
        with open(file_name, 'r') as trades_csv:
            trade_reader = csv.DictReader(trades_csv, delimiter=',')

            for tr in trade_reader:
                buying = tr["Action"] == "B"
                trade = Trade(tr["TIME"], tr["SYMBOL"], buying, 
                        float(tr["PRICE"]), int(tr["QUANTITY"]))
                
                self.process_trade(trade)
    
    # NEED TO UPDATE FOR SPECIFIC FILES
    def process_df(self, df):
            for i, tr in df.iterrows():
                buying = tr["Action"] == 'B'
                trade = Trade(tr["Date/Time"], tr["Symbol"], buying, 
                        float(tr["T. Price"]), int(tr["Quantity"]), 
                        float(tr['Comm/Fee']))
                
                self.process_trade(trade)
        
    def print_closed_trades(self):
        for ct in self._closed_trades:
            print(ct)

    def get_pnl(self):
        return self._pnl

    def get_copy_of_closed_trades(self):
        ''' returns shallow copy of closed trades '''
        return self._closed_trades[:]

class Trade():
    def __init__(self, time, symbol, buying, price, quantity, comm):
        self.time = time
        self.symbol = symbol
        self.buying = buying
        self.price = price
        self.quantity = quantity
        self.comm = comm

class ClosedTrade():
    def __init__(self, open_t, close_t, symbol, quantity, pnl, bought_first,
            open_p, close_p, comm_tot):
        self.open_t = open_t
        self.close_t = close_t
        self.symbol = symbol
        self.quantity = quantity
        self.pnl = pnl
        self.bought_first = bought_first
        self.open_p = open_p
        self.close_p = close_p
        self.comm_tot = comm_tot

    def __str__(self):
        s = "{},{},{},{},{:.2f},{},{},{:.2f},{:.2f}"
        s = s.format(
                self.open_t,
                self.close_t,
                self.symbol,
                self.quantity,
                self.pnl,
                "B" if self.bought_first else "S",
                "S" if self.bought_first else "B",
                self.open_p,
                self.close_p,
                self.comm_tot
                )
        return s
        
    def to_df(self):
        df = pd.DataFrame(
            [[   self.open_t,
                self.close_t,
                self.symbol,
                self.quantity,
                self.pnl,
                "B" if self.bought_first else "S",
                "S" if self.bought_first else "B",
                self.open_p,
                self.close_p,
                self.comm_tot
            ]], 
            columns = ['Open_Date','Close_Date','Symbol','Quantity', 'Pnl',
                       'OpenAct','CloseAct','Open_Price','Close_Price','Comm_Tot']
        )
        
        return df
        
        

        
if __name__ == "__main__":
    tm = TradeManager(store_trades=False, print_trades=True)
    print(("OPEN_TIME,CLOSE_TIME,SYMBOL,QUANTITY,PNL,"
            "OPEN_SIDE,CLOSE_SIDE,OPEN_PRICE,CLOSE_PRICE"))
    tm.process_csv(sys.argv[1])
    print("{:.2f}".format(tm.get_pnl()))