import pytest
import importlib  
main_file = importlib.import_module("P1-AnalyzeTrades_f_buildmodel_func")

def test_gini_1():
    a = [ 0, 1, 2]
    b = [ 0, 1, 2]
    assert main_file.gini_sklearn(a,b) == 1 , 'gini for equal indices not working' 