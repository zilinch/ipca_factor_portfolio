import pandas as pd
import numpy as np
from scipy.linalg import inv
from datetime import datetime
import sys
import argparse
from multiprocessing import Pool, Lock


from PortOpt_factor.optimizer import pyport
from logger import ErrorLogger
from ipca_utils import IPCA_factor


def init_pool_lock(l):
    global lock
    lock = l

def ipca_step_t_wrapper(args):
    return ipca_step_t(*args)


def ipca_step_t(t, df_ipca, window_size, unique_dates, K, charateristics, logger, res_fp):
    
    window_dates = unique_dates[t-window_size:t]
    cur_date = unique_dates[t]
    print (f'======Progress: {t}: {cur_date}======')

    # calculate ipca
    try:
        Gamma, factors, y_next, X_next, X_last, _ = IPCA_factor(df_ipca, t, window_dates, cur_date, charateristics, K, logger)
    except:
        return None
    
    V_t = inv(Gamma.T @ X_last.T @ X_last @ Gamma) @ Gamma.T @ X_last.T
    Sigma_t= np.cov(factors, rowvar = True)
    mu_t = np.array(np.mean(factors, axis=1))

    # Grid search for regularization terms
    g1 = np.exp(np.linspace( np.log(1e-6),np.log(5),10))
    g2 = np.exp(np.linspace( np.log(1e-6),np.log(5),10))
    lambda_pairs = [(l1, l2) for l1 in g1 for l2 in g2]
    
    max_ret_t = -1
    best_l1, best_l2 = None, None
  
    for l1, l2 in lambda_pairs:
    
        OptimalPortfolioWeights_t = pyport.portfolio_optimization(
            meanVec=np.array(mu_t),
            sigMat=np.array(Sigma_t),
            retTarget=0,
            longShort=0.2,
            maxAlloc=0.08,
            lambda_l1=l1,
            lambda_l2=l2,
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

        ret_t = (inv(Gamma.T @ X_next.T @ X_next @ Gamma) @ Gamma.T @ X_next.T @ y_next).T @ OptimalPortfolioWeights_t
        
        if ret_t > max_ret_t:
            max_ret_t = ret_t
            best_l1, best_l2 = l1, l2
                
            
    print (t, max_ret_t, best_l1, best_l2)
    return     
        
        
        
def main():
    
    if len(sys.argv) < 2:
        print ("Usage: run_job.py --window=200")
        sys.exit(1)
        
    fn = "data/factor_data_new.p"
    df = pd.read_pickle(fn)
    cols_to_drop = ["isin", "cusip", "sedol", "gics", "sic", "naics", "excntry", "ret_exc_lead1m", "ret_exc", 'ret_local']
    df_ipca = df.drop(cols_to_drop, axis=1)
    # remove all rows/stocks that have missing values for next month return
    df_ipca = df_ipca.dropna(subset=['ret_local_lead1m'])
    
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--window', type=int, help='Size of the window')
        args = parser.parse_args()
        window_size = args.window
    except:
        raise ValueError("Invalid integer number for window size")
    
    K = 6 #num of principle components
    charateristics = df_ipca.columns[3:] #list of characteristics
    unique_dates = sorted(df_ipca['eom'].unique()) #unique dates
    T = len(unique_dates)
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    log_fp = "logs/"+f"{current_date}-w{window_size}-log-error.txt"
    res_fp = "results/"+f"{current_date}-w{window_size}-results.txt"
    logger = ErrorLogger(log_filename=log_fp) #each node/window has separate log file
    
    args_list = [(t, df_ipca, window_size, unique_dates, K, charateristics, logger, res_fp) for t in range(window_size, T)]
    
    lock = Lock()
    with Pool(initializer=init_pool_lock, initargs=(lock,)) as pool:
        pool.map(ipca_step_t_wrapper, args_list)




if __name__ == "__main__":
    main()
    
    
    
    
    
    
   