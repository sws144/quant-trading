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
