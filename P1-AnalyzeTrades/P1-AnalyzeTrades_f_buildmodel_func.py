# functions for step f
import numpy as np

def gini(actual, pred):
    assert (len(actual) == len(pred))
    allv = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    allv = allv[np.lexsort((allv[:, 2], -1 * allv[:, 1]))]
    totalLosses = allv[:, 0].sum()
    giniSum = allv[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

# can swap in
def gini_xgb(predictions, truth):
    truth = truth.get_label()
    return 'gini', -1.0 * gini(truth, predictions) / gini(truth, truth)

# can swap in
def gini_lgb(truth, predictions):
    score = gini(truth, predictions) / gini(truth, truth)
    return 'gini', score, True

def gini_sklearn(truth, predictions):
    return gini(truth, predictions) / gini(truth, truth)

# legacy TODO delete

    # https://www.kaggle.com/eikedehling/tune-and-compare-xgb-lightgbm-rf-with-hyperopt

    # def gini(solution, submission):  # actual, expected
    #     """expects 2 lists"""                                       
    #     df = sorted(zip(solution, submission),    
    #             key=lambda x: x[1], reverse=True) # still a list, sorted by y_pred
    #     random = [float(i+1)/float(len(df)) for i in range(len(df))] # uniform percentiles             
    #     totalPos = np.sum([x[0] for x in df]) # sum of actual results                                      
    #     cumPosFound = np.cumsum([x[0] for x in df]) # list of cumulative actual                               
    #     Lorentz = [float(x)/totalPos for x in cumPosFound] # curve                        
    #     Gini = [l - r for l, r in zip(Lorentz, random)] # slice of diff from Lorenz and random                          
    #     return np.sum(Gini)   
