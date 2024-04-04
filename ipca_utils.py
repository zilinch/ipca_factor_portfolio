from typing import List
import sys
from ipca import InstrumentedPCA

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
import gc
import warnings
warnings.filterwarnings('ignore')

def impute_w_median(
    df: pd.DataFrame,
    characteristics: List
):
    
    df_temp = df.set_index(['eom', 'id'])
    df_chars = df_temp[characteristics]
    medians_by_time = df_chars.groupby('eom').transform('median')
    chars_imputed = df_chars.fillna(medians_by_time)
    df_temp.update(chars_imputed)
    df_imputed = df_temp.reset_index()
    return df_imputed
    

def normalize(
    df: pd.DataFrame,
    characteristics: List,
    scaler = 'standard'
):
    if scaler =='standard':
        scaler = StandardScaler()
    elif scaler == 'quantile':
        scaler = QuantileTransformer()
    elif scaler == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise NotImplementedError
    
    def scale_characteristics(group):
        group[characteristics] = scaler.fit_transform(group[characteristics])
        return group
    
    df_temp = df.set_index(['eom', 'id'])
    normalized_df = df_temp.groupby(level='eom', as_index=False).apply(scale_characteristics)
    df_new = normalized_df.reset_index(['eom', 'id'])
    return df_new
    


def IPCA_factor(
    window_data, 
    characteristics, 
    K
):        
    # Get the last date data and train data from the window data
    last_date = max(window_data['eom'].values)
    last_win_data = window_data[window_data['eom'] == last_date].copy()
    last_win_data.set_index('id', inplace=True)
    train_data = window_data[window_data['eom'] != last_date].copy()
    
    # Only use columns with limited NaN
    nan_threshold = 0.05
    chars_to_keep = []
    for char in characteristics:
        if train_data[char].isna().mean() <= nan_threshold:
            chars_to_keep.append(char)
    r_t = last_win_data['ret_local_lead1m']
    excess_r_t = last_win_data['ret_exc_lead1m']
    X_last = last_win_data[chars_to_keep]
    
    # Drop rows where lead 1m return (label) is missing
    train_data = train_data.dropna(subset=["ret_local_lead1m"])
    
    # Prepare data for IPCA model
    train_data.set_index(['id', 'eom'], inplace=True) # this format is required for Kelly's IPCA module
    y = train_data['ret_local_lead1m'] #lead return
    X = train_data[chars_to_keep]
    
    ## Fill NA with median
    X = X.fillna(X.median())
    X_last = X_last.fillna(X.median()) # Don't use future info?
    
    try:
        regr = InstrumentedPCA(n_factors=K, intercept=False, max_iter=400, iter_tol=1e-4)
        regr = regr.fit(X=X, y=y, quiet = True)
        Gamma, Factors = regr.get_factors(label_ind=True)
        # Free up memory immediately
        del train_data
        del last_win_data
        gc.collect()
    except:
        # Free up memory immediately
        del train_data
        del last_win_data
        gc.collect()

        raise ValueError(f"Unable to fit IPCA model with K = {K}")
    
    return Gamma, Factors, r_t, excess_r_t, X_last