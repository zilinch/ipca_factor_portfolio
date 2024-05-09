import numpy as np
import pandas as pd
from scipy.linalg import inv
import sys
import os
from multiprocessing import Pool
import itertools

from factor_port_opt import portfolio_optimization_tc

window_size = 240 # Just decides which index to start
longShort = 0.2
backtest_result_path = "results/backtest_weights/LongShort_0.2/"

lambda_l1_list = [0.001, 0.0001, 0.00001, 0.000001]
lambda_l2_list = [0.001, 0.0001, 0.00001, 0.000001]
lambda_l3_list = [0]

def run_factor_port_backtest(args):
    # Unpack arguments
    lambda_l1, lambda_l2, lambda_l3 = args
    
    save_filename = f"asset_weights_one_{lambda_l1}_{lambda_l2}_{lambda_l3}.csv"
    
    # Skip if results already exists
    if os.path.isfile(f"{backtest_result_path}{save_filename}"):
        return
    
    # load data
    df_ipca = pd.read_pickle("data/kelly_data_without_nanocap.p")
    df_ipca = df_ipca[["id","eom","ret_local","ret_exc","prc"]]
    # impute and normalize
    df_ipca.sort_values(by='eom', inplace=True, ignore_index=True)

    unique_dates = sorted(df_ipca['eom'].unique()) #unique dates
    T = len(unique_dates)

    test_period = len(unique_dates) - window_size

    backtest_cost = df_ipca.copy()
    backtest_cost["transaction_cost"] = 0.0085 / backtest_cost.prc
    backtest_cost = backtest_cost[["id","eom","transaction_cost"]]
    backtest_cost = backtest_cost.pivot_table(index="eom",columns="id",values="transaction_cost")
    backtest_cost.index.name = None
    backtest_cost.columns.name = None
    backtest_cost.columns = backtest_cost.columns.astype(str)

    backtest_ret = df_ipca.copy()
    backtest_ret = backtest_ret[["id","eom","ret_local"]]
    backtest_ret = backtest_ret.pivot_table(index="eom",columns="id",values="ret_local")
    backtest_ret.index.name = None
    backtest_ret.columns.name = None
    backtest_ret.columns = backtest_ret.columns.astype(str)
    backtest_lead_ret = backtest_ret.shift(-1)

    backtest_weights = pd.DataFrame(columns=sorted(df_ipca.id.unique().astype(str)),
                                    index=unique_dates[window_size-1:window_size+test_period])
    backtest_weights.columns = backtest_weights.columns.astype(str)
    backtest_weights = backtest_weights.fillna(0)

    for t in range(window_size-1, window_size - 1 + test_period):
        # t+1 is date to predict, t is current date, t-1 is previous date
        if t%40 == 0:
            print(f"========Progress for L1={lambda_l1}, L2={lambda_l2}, L3={lambda_l3}========")
            print(f"{t} / {T} completed, now at {str(unique_dates[t])}")
        date_to_predict, Gamma, Factors, r_t, excess_r_t, X_last \
            = pd.read_pickle(f'results/IPCA_intermediates_v1/predicting_{unique_dates[t+1]}.pickle')
        if t == window_size-1:
            # Initialize variables
            K = Factors.shape[0]

            # Update previous weights
            # 0 weight at time 0
            w_prev = np.zeros(len(r_t))
        else:
            w_prev = backtest_weights[r_t.index.astype(str)].loc[unique_dates[t-1]]
            # Update w_prev to reflect price move from t-1 to t
            w_prev = (w_prev * backtest_ret.loc[unique_dates[t]]).sort_index().fillna(0)
            # By doing this, there is error in tracking TC because we're not tracking every asset, but should be mostly accurate
            w_prev = w_prev.loc[r_t.index.astype(str)]
            # Normalize w_prev to leverage amount
            w_prev = w_prev / sum(abs(w_prev)) * (1 + longShort*2)

        # Get costVec
        costVec = backtest_cost[r_t.index.astype(str)].loc[unique_dates[t]]
        costVec = np.array(costVec)

        # Portfolio optimization
        V_t = inv(Gamma.T @ X_last.T @ X_last @ Gamma) @ Gamma.T @ X_last.T
        # Regularize V matrix
        #reg_mat = np.zeros_like(V_t)
        #reg_mat[:K, :K] = np.eye(K)*1e-4
        #V_t += reg_mat

        Sigma_t = np.cov(Factors, rowvar = True)
        mu_t = np.array(np.mean(Factors, axis=1))
        optimal_w_f, var_port = portfolio_optimization_tc(
            meanVec   = mu_t,
            sigMat    = Sigma_t,
            V         = V_t.T,
            w_prev    = w_prev,
            costVec   = costVec,
            longShort = longShort,
            maxAlloc  = 0.2,
            lambda_l1 = lambda_l1,
            lambda_l2 = lambda_l2,
            lambda_l3 = lambda_l3,
        )
        # Change optimal_w_f into a 1xK matrix
        optimal_w_f = optimal_w_f.reshape(-1, 1)

        w_t = (np.array(V_t.T) @ optimal_w_f).flatten()
        w_t = w_t / sum(abs(w_t)) * (1 + longShort*2)
        w_t = pd.Series(w_t, index=V_t.columns).sort_index()
        w_t.index = w_t.index.astype(str)
        w_t.index.name = None
        backtest_weights.loc[unique_dates[t]] = w_t
        backtest_weights = backtest_weights.fillna(0)

    # Save results
    if not os.path.exists(backtest_result_path):
        os.makedirs(backtest_result_path)
    backtest_weights.to_csv(f"{backtest_result_path}{save_filename}", index=True)

    return
    

if __name__ == "__main__":
    product_args = itertools.product(lambda_l1_list, lambda_l2_list, lambda_l3_list)
    with Pool(4) as pool:
        results = pool.map(run_factor_port_backtest, product_args)
