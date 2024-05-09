import sys
import pandas as pd
import numpy as np
from scipy.linalg import inv
import pickle
import multiprocessing as mp
from multiprocessing import Pool, Lock
import os
import argparse
from datetime import datetime

from ipca_utils import IPCA_factor_v7
from logger import ErrorLogger


def init_pool_lock(l):
    global lock
    lock = l

def ipca_step_t_wrapper(args):
    return ipca_step_t(*args)


def ipca_step_t(t, window_size, df_ipca, unique_dates, K, characteristics, log_fp, ipca_fp):
    '''
    This function only runs IPCA model and stores the factors matrix and V matrix at time t.
    Portfolio optimization is put to another part of the program as it depends on t-1 weights
    '''   

    window_dates = unique_dates[t-window_size:t]
    mask = df_ipca['eom'].isin(window_dates)
    window_data = df_ipca[mask]
    date_to_predict = unique_dates[t]

    # Skip if already has result
    if os.path.isfile(f"{ipca_fp}predicting_{date_to_predict}.pickle"):
        print (f'====== Skipped: {t}: Date to predict: {date_to_predict} ======')
        return 0
    
    logger = ErrorLogger(log_filename=log_fp) #logger
    
    print (f'====== Started: {t}: Date to predict: {date_to_predict} ======')
    
    # Fit ipca
    try:
        Gamma, Factors, r_t, excess_r_t, X_last = IPCA_factor_v7(window_data, characteristics, K)
    except Exception as e:
        with lock:
            logger.log_error(str(date_to_predict), e)
        return 0
    
    ipca_output_t = [date_to_predict, Gamma, Factors, r_t, excess_r_t, X_last]
    with open(f"{ipca_fp}predicting_{date_to_predict}.pickle", 'wb') as handle:
        pickle.dump(ipca_output_t, handle)

    print (f'====== Complete: {t}: Date to predict: {date_to_predict} ======')

    return 0


def main():
    
    # Load data
    df_ipca = pd.read_pickle("data/kelly_data_no_nanocap_quantile_transformed.p")
    # The pickle file's "eom" column is already in datetime.date format, this is just in case
    df_ipca.eom = pd.to_datetime(df_ipca.eom).dt.date
    print ("====== Finished loading data ======")
    # Store characteristics names
    characteristics = df_ipca.columns[10:] #list of characteristics
    df_ipca.sort_values(by='eom', inplace=True, ignore_index=True)

    window_size = 240 # in minths
    K = 6 # num of principle components
    unique_dates = sorted(df_ipca['eom'].unique()) # unique dates
    T = len(unique_dates)
   
    current_date = datetime.now().strftime('%Y-%m-%d')
    log_fp = "logs/"+f"{current_date}-w{window_size}-log-error.txt"
    ipca_fp = f"results/IPCA_intermediates_v7/"
    
    if not os.path.exists(ipca_fp):
        os.makedirs(ipca_fp)
    
    args_list = [(t, window_size, df_ipca, unique_dates, K, characteristics, log_fp, ipca_fp) \
                        for t in range(window_size, T)]
    
    lock = Lock()
    # 6-8 processes looks good for 32GB of RAM
    with Pool(processes=4, initializer=init_pool_lock, initargs=(lock,)) as pool:
        pool.map(ipca_step_t_wrapper, args_list)

    return 0



if __name__ == '__main__':
    main()