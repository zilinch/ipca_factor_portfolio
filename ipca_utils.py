import pandas as pd
from typing import List
from ipca import InstrumentedPCA


def IPCA_impute(
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
    
    # remove all rows/stocks that have missing values for next month return
    df = df.dropna(subset=['ret_local_lead1m'])

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
    
    return df_imputed, characteristics



def IPCA_factor(df, selected_dates, date_to_predict, characteristics, K):
    """
    Apply Instrumented PCA (IPCA) on a data frame to extract K principal components on a rolling window bais.
    This function uses the Python implementation of the Instrumtented Principal Components Analysis 
    framework by Kelly, Pruitt, Su. The paper can be found at:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2983919

    Parameters:
    ----------
    df:
        Pandas dataframe containing the return data and the characteristics
    selected_dates:
        All unique datetime within the current rolling window
    date_to_predict:
        The next date to predict
    characteristics:
        List of characteristics used for computing ipca
    K:
        Number of principal components

    """
    

    # Filter the data to include only rows within the selected date range
    window_data = df[df['eom'].isin(selected_dates)]
    window_data_imputed, characteristics_to_keep = IPCA_impute(window_data, characteristics, threshold=30)
    
    # Get the last date data from the window data
    last_data = window_data_imputed[window_data_imputed['eom'] == selected_dates[-1]]
    last_ids = last_data['id'].unique()

    # Get the predict/next data but only keep stocks & characteristics appeared in last data
    next_data = df[df['eom'] == date_to_predict]
    next_data = next_data[next_data['id'].isin(last_ids)]  ###dont do this + drop stocks with no return
    next_data_aligned = next_data[last_data.columns]
    next_data_imputed, _ = IPCA_impute(next_data_aligned, characteristics_to_keep, drop_characteristics=False)
    

    y_next = next_data_imputed['ret_local']
    X_next = next_data_imputed[characteristics_to_keep]
    X_last = last_data[characteristics_to_keep]
    
    assert X_next.shape[1] == X_last.shape[1], "Charateristics used for prediction has to be the same\
                    as the characteristics used for training"
    
    # IPCA model
    window_data_imputed.set_index(['id', 'eom'], inplace=True)
    y = window_data_imputed['ret_local'] #cur return
    X = window_data_imputed[characteristics_to_keep]
    regr = InstrumentedPCA(n_factors=K, intercept=False, iter_tol=1e-4)
    regr = regr.fit(X=X, y=y, quiet = True)
    Gamma, Factors = regr.get_factors(label_ind=True)
    
    return Gamma, Factors, y_next, X_next, X_last, last_ids
