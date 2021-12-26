# %% [markdown]
##  ## E: Feature Engineering

# %%
## imports

import pandas as pd
import numpy as np

# for na pipeline
import warnings
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import TransformerMixin # for custom transformers

from joblib import dump, load

# %%
## read in data

df_XY = pd.read_csv('output/c_resulttradewattr.csv')

# %%
##  get_feature_names function 
# https://johaupt.github.io/scikit-learn/tutorial/python/data%20processing/ml%20pipeline/model%20interpretation/columnTransformer_feature_names.html
def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """
    # Remove the internal helper function
    #check_is_fitted(column_transformer)
    
    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return [i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [f for f in column]

        return [f for f in trans.get_feature_names()]
    
    ### Start of processing
    feature_names = []
    
    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = column_transformer.transformers_
    
    
    for name, trans, column in l_transformers: 
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    
    return feature_names

# %%
## custom transformers
class Numerizer(TransformerMixin):
    import pandas as pd
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Y = X.apply(pd.to_numeric, errors='coerce')
        return Y
    
class StringTransformer(TransformerMixin):
    import pandas as pd
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Y = pd.DataFrame(X).astype('string')
        return Y

# %%
## create na pipeline

# update columns headers to clean up
df_XY.columns = list(
    pd.Series(df_XY.columns).astype(str).str.replace(' ','_', regex=True)\
        .str.upper().str.strip().str.replace('/','_').str.replace('*','_')
)

# start with numeric, utilizng explore data before
numeric_features = df_XY.select_dtypes(include=np.number).columns.tolist()
numeric_features = numeric_features + [
    '%_TO_STOP',
    '%_TO_TARGET',
    'GROWTH_0.5TO0.75',
    'ROIC_(BW_ROA_ROE)',
    'IMPLIED_P_E',
    'YEARS_TO_NORMALIZATION',
]
numeric_features = list(set(numeric_features))

numeric_transformer = Pipeline(
    steps=[
        ('numerizer', Numerizer()),          
        ('imputer', SimpleImputer(strategy='constant', fill_value = -1)),
    ]
)
categorical_transformer = Pipeline(
    steps=[          
        ('imputer', SimpleImputer(strategy='constant', fill_value = '_NA_')),
        ('stringtransformer', StringTransformer()),    
    ]
)

# numerical

# categorical_features = ['embarked', 'sex', 'pclass']
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))])

categorical_features = list(set(df_XY.columns).difference(set(numeric_features)))

preprocessor_na = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features) ,
        ('cat', categorical_transformer, categorical_features)          
    ], 
    # remainder = 'passthrough' # not needed anymore
)

XY_imputed = preprocessor_na.fit_transform(df_XY)

columns =get_feature_names(preprocessor_na)

df_XY_imputed = pd.DataFrame(XY_imputed,columns=columns).convert_dtypes()


# %% 
# create target 

df_XY_imputed['PCT_RET_FINAL'] = (
    df_XY_imputed['PNL'] / ( df_XY_imputed['OPEN_PRICE'] *  df_XY_imputed['QUANTITY']) 
)

# %% 
# TODO create moving avg



# %% 
# Final columns

print(df_XY_imputed.columns)

# %%
## check no na's left in numerical

try: 
    assert df_XY_imputed[numeric_features].isna().sum().sum() == 0 , 'NAs remain in numerical'
except:
    print('NAs remain in numerical')

# %%
## test against api spec

import yaml
from yaml import Loader

with open("data-tests/_apispecs.yaml") as f:
    data = yaml.load(f, Loader=Loader)

# %%
## save api spec to html

import os

os.system('python swagger_yaml_to_html.py < data-tests/_apispecs.yaml > templates/api.html')


# %%
## save results

df_XY_imputed.to_csv('output/e_resultcleaned.csv')


# %%
## save imputer

dump(preprocessor_na,'output/e_preprocessor_na.joblib')

# %%
