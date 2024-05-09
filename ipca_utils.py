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

def IPCA_factor_v2(
    window_data, 
    characteristics
):        
    # Get the last date data and train data from the window data
    last_date = max(window_data['eom'].values)
    last_win_data = window_data[window_data['eom'] == last_date].copy()
    last_win_data.set_index('id', inplace=True)
    train_data = window_data[window_data['eom'] != last_date].copy()
    
    # Only use columns with limited NaN
    nan_threshold = 0.1
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
    
    # Fill NA with median
    X = X.fillna(X.median())
    X_last = X_last.fillna(X.median()) # Don't use future info?
    
    try:
        K = int(X_last.shape[1] * 0.2)
        regr = InstrumentedPCA(n_factors=K, intercept=False, max_iter=400, iter_tol=1e-5)
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

def IPCA_factor_v3(
    window_data, 
    characteristics,
    K
):        
    # Use 1000 stocks with the largest median market cap
    median_market_cap = window_data[["id","market_equity"]].groupby(by="id").median()
    index_large_cap = median_market_cap.market_equity.nlargest(1000).index.astype(str)
    window_data = window_data[window_data.id.astype(str).isin(index_large_cap)]

    # Get the last date data and train data from the window data
    last_date = max(window_data['eom'].values)
    last_win_data = window_data[window_data['eom'] == last_date]
    last_win_data.set_index('id', inplace=True)
    train_data = window_data[window_data['eom'] != last_date]

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

    # Quantile transformation with normal output
    data_transformer = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=0)
    X_trans = pd.DataFrame(data_transformer.fit_transform(X))
    X_trans.index = X.index
    X_trans.columns = X.columns

    X_last_trans = pd.DataFrame(data_transformer.transform(X_last))
    X_last_trans.index = X_last.index
    X_last_trans.columns = X_last.columns

    ## Fill NA with median
    X_trans = X_trans.fillna(X_trans.median())
    X_last_trans = X_last_trans.fillna(X_trans.median()) # Don't use future info?

    
    
    try:
        regr = InstrumentedPCA(n_factors=K, intercept=False, max_iter=400, iter_tol=1e-4)
        regr = regr.fit(X=X_trans, y=y, quiet = True)
        Gamma, Factors = regr.get_factors(label_ind=True)
    except:
        raise ValueError(f"Unable to fit IPCA model with K = {K}")
    
    return Gamma, Factors, r_t, excess_r_t, X_last_trans


def IPCA_factor_v4(
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
    nan_threshold = 0.1
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


def IPCA_factor_v6(
    window_data, 
    characteristics, 
    K
):
    # Determine which feature columns to keep based on missing-ness
    nan_threshold = 0.1
    chars_to_keep = []
    for char in characteristics:
        if window_data[char].isna().mean() <= nan_threshold:
            chars_to_keep.append(char)

    # Only keep id, eom, X and y columns, and r_t columns (to save to pickle)
    window_data = window_data[["id", "eom", "ret_local_lead1m", "ret_exc_lead1m"] + chars_to_keep]

    # Fill Nan, missing at beginning and end will be dropped at train_data
    window_data = window_data.sort_values(['id', 'eom'], ascending=[True, True])
    window_data = window_data.groupby(by='id').apply(lambda group: group.interpolate(method='linear'))
    window_data = window_data.reset_index(drop=True)

    # Get the last date data and train data from the window data
    last_date = max(window_data['eom'].values)
    last_win_data = window_data[window_data['eom'] == last_date].copy()
    last_win_data.set_index('id', inplace=True)
    train_data = window_data[window_data['eom'] != last_date].copy()
    
    # Save data to returns
    # Remove assets (don't trade) in X_last if any required factor is missing
    X_last = last_win_data[chars_to_keep]
    X_last = X_last.dropna()
    r_t = last_win_data['ret_local_lead1m']
    excess_r_t = last_win_data['ret_exc_lead1m']
    
    # Drop rows where lead 1m return (label) is missing
    # After willing nan for all columns, this is useless
    #train_data = train_data.dropna(subset=["ret_local_lead1m"])
    
    # Prepare data for IPCA model
    train_data.set_index(['id', 'eom'], inplace=True) # this format is required for Kelly's IPCA module
    train_data = train_data.dropna()
    y = train_data['ret_local_lead1m'] #lead return
    X = train_data[chars_to_keep]
    
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


def IPCA_factor_v7(
    window_data, 
    characteristics, 
    K
):
    # Determine which feature columns to keep based on missing-ness
    nan_threshold = 0.1
    chars_to_keep = []
    for char in characteristics:
        if window_data[char].isna().mean() <= nan_threshold:
            chars_to_keep.append(char)

    # Only keep id, eom, X and y columns, and r_t columns (to save to pickle)
    window_data = window_data[["id", "eom", "ret_local_lead1m", "ret_exc_lead1m"] + chars_to_keep]

    ## Fill Nan, missing at beginning and end will be dropped at train_data
    #window_data = window_data.sort_values(['id', 'eom'], ascending=[True, True])
    #window_data = window_data.groupby(by='id').apply(lambda group: group.interpolate(method='linear'))
    #window_data = window_data.reset_index(drop=True)

    # Get the last date data and train data from the window data
    last_date = max(window_data['eom'].values)
    last_win_data = window_data[window_data['eom'] == last_date].copy()
    last_win_data.set_index('id', inplace=True)
    train_data = window_data[window_data['eom'] != last_date].copy()
    
    # Save data to returns
    # Remove assets (don't trade) in X_last if any required factor is missing
    X_last = last_win_data[chars_to_keep]
    X_last = X_last.dropna()
    r_t = last_win_data['ret_local_lead1m']
    excess_r_t = last_win_data['ret_exc_lead1m']
    
    # Prepare data for IPCA model
    train_data.set_index(['id', 'eom'], inplace=True) # this format is required for Kelly's IPCA module
    train_data = train_data.dropna() # Train data already contains only needed columns
    y = train_data['ret_local_lead1m'] # lead return
    X = train_data[chars_to_keep]
    
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


def IPCA_factor_v8(
    window_data, 
    characteristics, 
    K
):
    # Determine which feature columns to keep based on missing-ness
    nan_threshold = 0.4
    chars_to_keep = []
    for char in characteristics:
        if window_data[char].isna().mean() <= nan_threshold:
            chars_to_keep.append(char)

    # Only keep id, eom, X and y columns, and r_t columns (to save to pickle)
    window_data = window_data[["id", "eom", "ret_local_lead1m", "ret_exc_lead1m"] + chars_to_keep]

    ## Fill Nan, missing at beginning and end will be dropped at train_data
    #window_data = window_data.sort_values(['id', 'eom'], ascending=[True, True])
    #window_data = window_data.groupby(by='id').apply(lambda group: group.interpolate(method='linear'))
    #window_data = window_data.reset_index(drop=True)

    # Get the last date data and train data from the window data
    last_date = max(window_data['eom'].values)
    last_win_data = window_data[window_data['eom'] == last_date].copy()
    last_win_data.set_index('id', inplace=True)
    train_data = window_data[window_data['eom'] != last_date].copy()
    
    # Save data to returns
    # Remove assets (don't trade) in X_last if any required factor is missing
    X_last = last_win_data[chars_to_keep]
    X_last = X_last.dropna()
    r_t = last_win_data['ret_local_lead1m']
    excess_r_t = last_win_data['ret_exc_lead1m']
    
    # Prepare data for IPCA model
    train_data.set_index(['id', 'eom'], inplace=True) # this format is required for Kelly's IPCA module
    train_data = train_data.dropna() # Train data already contains only needed columns
    y = train_data['ret_local_lead1m'] # lead return
    X = train_data[chars_to_keep]
    
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
