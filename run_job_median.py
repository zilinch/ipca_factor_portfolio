import sys
import pandas as pd
import numpy as np
from scipy.linalg import inv
from multiprocessing import Pool, Lock
import os
import argparse
from datetime import datetime

from PortOpt_factor.optimizer import pyport
from ipca_utils import impute_w_median, IPCA_factor
from logger import ErrorLogger


def init_pool_lock(l):
    global lock
    lock = l

def ipca_step_t_wrapper(args):
    return ipca_step_t(*args)


def ipca_step_t(t, window_size, df_ipca, unique_dates, K, characteristics, log_fp, res_fp, wts_fp):
    
    window_dates = unique_dates[t-window_size:t]
    mask = df_ipca['eom'].isin(window_dates)
    window_data = df_ipca[mask]
    date_to_predict = unique_dates[t]
    
    logger = ErrorLogger(log_filename=log_fp) #logger
    
    print (f'======Progress: {t}: {date_to_predict}======')
    
    # calculate ipca
    try:
        Gamma, Factors, r_t, excess_r_t, X_last = IPCA_factor(window_data, characteristics, K)
    except Exception as e:
        with lock:
            logger.log_error(date_to_predict, e)
        return 0

    # regularization
    V_t = inv(Gamma.T @ X_last.T @ X_last @ Gamma) @ Gamma.T @ X_last.T
    reg_mat = np.zeros_like(V_t)
    reg_mat[:K, :K] = np.eye(K)*1e-04
    V_t += reg_mat

    Sigma_t = np.cov(Factors, rowvar = True)
    mu_t = np.array(np.mean(Factors, axis=1))

    # Grid search for regularization terms
    g1 = np.exp(np.linspace( np.log(1e-6),np.log(5),10))
    g2 = np.exp(np.linspace( np.log(1e-6),np.log(5),10))
    
    try:
        OptimalPortfolioWeights_t = pyport.portfolio_optimization(
            meanVec=np.array(mu_t),
            sigMat=np.array(Sigma_t),
            retTarget=0,
            longShort=0.2,
            maxAlloc=0.08,
            lambda_l1=g1[0],
            lambda_l2=g2[0],
            riskfree=0,
            assetsOrder=None,
            maxShar=1,
            factor=np.array(V_t.T),
            turnover=None,
            w_pre=None,
            individual=False,
            exposure_constrain=0,
            w_bench=None,
            factor_exposure_constrain=None,
            U_factor=None,
            general_linear_constrain=None,
            U_genlinear=0,
            w_general=None,
            TE_constrain=0,
            general_quad=0,
            Q_w=None,
            Q_b=None,
            Q_bench=None
        )[0].reshape(-1, 1)
    except Exception as e:
        with lock:
            logger.log_error(date_to_predict, e)
        return 0
    
    ret_t = (V_t @ r_t).T @ OptimalPortfolioWeights_t
    excess_ret_t = (V_t @ excess_r_t).T @ OptimalPortfolioWeights_t
    w_individual = np.dot(np.array(V_t.T), OptimalPortfolioWeights_t).flatten()
    pos_weight = w_individual[w_individual > 0].sum()
    
    # print (f'======Max_return: {t}: {ret_t}======')    
    
    df_weights = pd.DataFrame({
        'ID': X_last.index,
        'Weights': w_individual}) 
    df_weights.to_csv(f"{wts_fp}{date_to_predict}.csv", index=False)              
    
    df_results = pd.DataFrame([[date_to_predict, ret_t[0], excess_ret_t[0], pos_weight]],
                          columns=['Date', 'P_Return', 'P_Excess_Return', 'Sum_Positive_Weights'])
    
    with lock:
        df_results.to_csv(res_fp, mode='a', index=False, header=not os.path.exists(res_fp))
    return  0   


def main():
    
    #load data
    df_ipca = pd.read_csv('/gpfs/home/zilinchen/ipca_factor_portfolio/data/factor_data_qnormed.csv')
    print ("=======Finish load data======")
    # impute and normalize
    characteristics = df_ipca.columns[6:] #list of characteristics
    df_ipca.sort_values(by='eom', inplace=True, ignore_index=True)

    window_size = 240
    K = 6 #num of principle components
    unique_dates = sorted(df_ipca['eom'].unique()) #unique dates
    T = len(unique_dates)
   
    current_date = datetime.now().strftime('%Y-%m-%d')
    log_fp = "logs/"+f"{current_date}-w{window_size}-log-error.txt"
    res_fp = "results/"+f"{current_date}-w{window_size}-results.csv"
    wts_fp = "results/"+f"{current_date}-w{window_size}/"
    
    if not os.path.exists(wts_fp):
        os.makedirs(wts_fp)
    
    args_list = [(t, window_size, df_ipca, unique_dates, K, characteristics, log_fp, res_fp, wts_fp) \
                        for t in range(window_size, T)]
    
    
    lock = Lock()
    with Pool(initializer=init_pool_lock, initargs=(lock,)) as pool:
        pool.map(ipca_step_t_wrapper, args_list)

    return 0



if __name__ == '__main__':
    main()