import pytest
import importlib 
import pandas as pd 

main_file = importlib.import_module("tradehelper")

def test_trademanager_1():
    """[basic pnl calc with commision test]
    """
    test_df = pd.read_csv('data-tests/tradelog1.csv')

    tm = main_file.TradeManager(store_trades=True, print_trades=False)

    tm.process_df(test_df)

    # list of trade objects
    complete_trades = tm.get_copy_of_closed_trades() 

    # pushed to dataframe
    df_complete_trades = pd.concat([x.to_df() for x in complete_trades]).reset_index(drop=True)

    assert df_complete_trades['Pnl'].sum().round(2) == -713.48 # calc'ed manualy in Excel
 
 
def test_2_corpact():
    """[basic pnl calc with commision test]
    """
    test_df = pd.read_csv('data-tests/tradelog2_corpact.csv')

    tm = main_file.TradeManager(store_trades=True, print_trades=False)

    tm.process_df(test_df)

    # list of trade objects
    complete_trades = tm.get_copy_of_closed_trades() 

    # pushed to dataframe
    df_complete_trades = pd.concat([x.to_df() for x in complete_trades]).reset_index(drop=True)

    assert df_complete_trades['Pnl'].sum().round(2) == -2 # calc'ed manualy in Excel
 