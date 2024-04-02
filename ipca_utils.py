from typing import List
import sys
sys.path.append('/gpfs/home/zilinchen/ipca_factor_portfolio/ipca/ipca/')
from ipca import InstrumentedPCA

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
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
    last_win_data = window_data[window_data['eom'] == last_date]
    last_win_data.set_index('id', inplace=True)
    train_data = window_data[window_data['eom'] != last_date]
    
    # Remove null columns from both last period and window data
    chars_to_drop = []
    null_columns_last = last_win_data.columns[last_win_data.isna().any()]
    null_columns_window = train_data.columns[train_data.isna().any()]
    for char in characteristics:
        if (char in null_columns_last) or (char in null_columns_window):
            chars_to_drop.append(char)
    chars_to_keep = [item for item in characteristics if item not in chars_to_drop]

    r_t = last_win_data['ret_local_lead1m']
    excess_r_t = last_win_data['ret_exc_lead1m']
    X_last = last_win_data[chars_to_keep]

    
    # IPCA model
    train_data.set_index(['id', 'eom'], inplace=True)
    y = train_data['ret_local_lead1m'] #lead return
    X = train_data[chars_to_keep]
    
    try:
        regr = InstrumentedPCA(n_factors=K, intercept=False, max_iter=400, iter_tol=1e-4)
        regr = regr.fit(X=X, y=y, quiet = True)
        Gamma, Factors = regr.get_factors(label_ind=True)
    except:
        raise ValueError(f"Unable to fit IPCA model with K = {K}")
    
    return Gamma, Factors, r_t, excess_r_t, X_last