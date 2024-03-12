import pandas as pd
from typing import List
from ipca import InstrumentedPCA


def IPCA_impute_w_mean(
    df: pd.DataFrame, 
    characteristics: List[str], 
    threshold: int = None,
    drop_characteristics: bool = True
):
    """
    Impute missing values for window data used in IPCA_factor

    Parameters
    ----------
    df:
        Pandas dataframe to impute
    characteristics:
        List of the characteristics
    threshold:
        Threshold (in percent) of missing data to drop a characteristics
    drop_characteristics:
        Bool indicator for dropping characteristics when imputing. Default True.
    
    """
    assert len(characteristics) > 0, "List of characteristics cannot be empty"

    assert not drop_characteristics or (drop_characteristics and threshold is not None), \
                        "When drop_characteristics is True, a threshold must be provided."


    # if drop_characteristics: remove all columns/characteristics 
    # that have missing values greater than the threshold
    if drop_characteristics:
        percent_missing = df.isnull().mean() * 100
        characteristics_to_drop = percent_missing[percent_missing > threshold].index
        df = df.drop(columns=characteristics_to_drop)
        characteristics_to_keep = [item for item in characteristics if item not in characteristics_to_drop]
        characteristics = characteristics_to_keep

    # fill the missing values at t with median across stock
    df_temp = df.set_index(['eom', 'id'])
    medians_by_time = df_temp.groupby(level=0).transform('median')
    df_imputed = df_temp.fillna(medians_by_time).reset_index()
    df_imputed_sorted = df_imputed.sort_values(by='eom')

    return df_imputed_sorted, characteristics



def IPCA_factor(df, step, window_dates, current_date, characteristics, K, logger):
    
    global lock
    
    # Filter the data to include only rows within the selected date range
    window_data = df[df['eom'].isin(window_dates)]
    window_data_imputed, characteristics_to_keep = IPCA_impute_w_mean(window_data, characteristics, threshold=50)
    
    # Get the last date data from the window data
    last_win_data = window_data_imputed[window_data_imputed['eom'] == window_dates[-1]]
    last_ids = last_win_data['id'].unique()
    

    # Get the predict/next data but only keep stocks & characteristics appeared in last data
    ft_data = df[df['eom'] == current_date]
    
    # only use characteristics that appear in the train data
    ft_data_aligned = ft_data[last_win_data.columns]
    ft_data_imputed, _ = IPCA_impute_w_mean(ft_data_aligned, characteristics_to_keep, drop_characteristics=False)
    

    y_next = ft_data_imputed['ret_local_lead1m']
    X_next = ft_data_imputed[characteristics_to_keep]
    X_last = last_win_data[characteristics_to_keep]
    
    assert X_next.shape[1] == X_last.shape[1], "Charateristics used for prediction has to be the same\
                    as the characteristics used for training"
    
    # IPCA model
    window_data_imputed.set_index(['id', 'eom'], inplace=True)
    y = window_data_imputed['ret_local_lead1m'] #lead return
    X = window_data_imputed[characteristics_to_keep]
    
  
    def get_ipca(X, y, K, step, date):
        while K >= 2:
            
            try:
                regr = InstrumentedPCA(n_factors=K, intercept=False, max_iter=500, iter_tol=1e-4, n_jobs= -1)
                regr = regr.fit(X=X, y=y, quiet = True)
                Gamma, Factors = regr.get_factors(label_ind=True)
                return Gamma, Factors
            except Exception as e:
                with lock:
                    logger.log_error(f'ValueError for ({step}, K={K}): {date}', e)
                K -= 1
        
        raise ValueError("Unable to fit the model with K >= 2")
    
    try:
        Gamma, Factors = get_ipca(X, y, K, step, current_date)
    except Exception as e:
        with lock:
            logger.log_error(f'ValueError for {step}: {current_date}', e)
        raise 
    
    return Gamma, Factors, y_next, X_next, X_last, last_ids