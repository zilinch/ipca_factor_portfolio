import numpy as np
import pandas as pd
from scipy.linalg import inv
import sys
import os
from multiprocessing import Pool
import itertools

from factor_port_opt import portfolio_optimization_cvx_tc

window_size = 240 # Just decides which index to start
longShort = 0.5
#maxExposure = 3
maxAlloc = 0.1

backtest_result_path = f"results/backtest_weights/ipca_v7/LongShort_{longShort}_MaxAlloc_{maxAlloc}_cvxopt/"
if not os.path.exists(backtest_result_path):
    os.makedirs(backtest_result_path)

lambda_l1_list = [0.0001]
lambda_l2_list = [0.0001]
lambda_l3_list = [0, 10, 100, 1000]
top_n_assets_list = [100]

def run_factor_port_backtest(args):
    # Unpack arguments
    lambda_l1, lambda_l2, lambda_l3, n_assets_to_trade = args
    n_assets_to_trade = int(n_assets_to_trade)
    
    save_filename = f"{n_assets_to_trade}_assets_sum_one_{lambda_l1}_{lambda_l2}_{lambda_l3}.csv"
    
    # load data
    df_ipca = pd.read_pickle("data/kelly_data_without_nanocap.p")
    df_ipca = df_ipca[["id","eom","ret_local","ret_exc","prc"]]
    # The pickle file's "eom" column is already in datetime.date format, this is just in case
    df_ipca.eom = pd.to_datetime(df_ipca.eom).dt.date
    df_ipca.sort_values(by='eom', inplace=True, ignore_index=True)

    unique_dates = sorted(df_ipca['eom'].unique()) # unique dates
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

    # If result file already exists, read and continue
    if os.path.isfile(f"{backtest_result_path}{save_filename}"):
        backtest_weights = pd.read_csv(f"{backtest_result_path}{save_filename}", index_col=0)
        backtest_weights.index = pd.to_datetime(backtest_weights.index).date
        # Find the index of the first row where sum of absolute weights is 0
        continue_date = backtest_weights.loc[(abs(backtest_weights).sum(axis=1) == 0)].index[0]
        df_ipca = df_ipca.query("eom >= @continue_date").reset_index(drop=True)
        df_ipca.sort_values(by='eom', inplace=True, ignore_index=True)
        #
        num_dates_left = len(df_ipca.eom.unique())
        t_shift = test_period - num_dates_left
        print(f"Skipped {t_shift} dates, starting at {unique_dates[window_size-1 + t_shift]}")
    else:
        t_shift = 0
        backtest_weights = pd.DataFrame(columns=sorted(df_ipca.id.unique().astype(str)),
                                    index=unique_dates[window_size-1:window_size+test_period])

    backtest_weights.columns = backtest_weights.columns.astype(str)
    backtest_weights = backtest_weights.fillna(0)
    

    for t in range(window_size-1 + t_shift, len(unique_dates)-1):
        # t+1 is date to predict, t is current date, t-1 is previous date
        if t%20 == 0:
            print(f"========Progress for L1={lambda_l1}, L2={lambda_l2}, L3={lambda_l3}========")
            print(f"{t} / {T} completed, now at {str(unique_dates[t])}")
        
        date_to_predict, Gamma, Factors, r_t, excess_r_t, X_last \
            = pd.read_pickle(f'results/IPCA_intermediates_v7/predicting_{unique_dates[t+1]}.pickle')
        X_last.index = X_last.index.astype(str)

        # Choose top n assets with largest market cap
        index_to_trade = X_last.market_equity.nlargest(n_assets_to_trade).index.astype(str)
        index_to_trade = index_to_trade.sort_values()
        X_last = X_last.loc[index_to_trade]
        
        if t == window_size-1:
            # Update previous weights
            # 0 weight at time 0
            w_prev = np.zeros(len(index_to_trade))
        else:
            w_prev = backtest_weights.loc[unique_dates[t-1]]
            # Update w_prev to reflect price move from t-1 to t
            w_prev = (w_prev * backtest_ret.loc[unique_dates[t]]).sort_index().fillna(0)
            # For assets that disappears, the transaction cost is constant because we have to clear our position
            # so we can simply ignore them in the optimization
            w_prev = w_prev.loc[index_to_trade.astype(str)]

        # Get costVec
        costVec = backtest_cost[index_to_trade.astype(str)].loc[unique_dates[t]]
        costVec = np.array(costVec)

        # Portfolio optimization
        V_t = inv(Gamma.T @ X_last.T @ X_last @ Gamma) @ Gamma.T @ X_last.T

        Sigma_t = np.cov(Factors, rowvar = True)
        mu_t = np.array(np.mean(Factors, axis=1))
        optimal_w_asset = portfolio_optimization_cvx_tc(
            meanVec   = mu_t,
            sigMat    = Sigma_t,
            V         = V_t.T,
            w_prev    = w_prev,
            costVec   = costVec,
            longShort = longShort,
            maxAlloc  = maxAlloc,
            lambda_l1 = lambda_l1,
            lambda_l2 = lambda_l2,
            lambda_l3 = lambda_l3,
        )

        w_t = pd.Series(optimal_w_asset, index=V_t.columns).sort_index()
        w_t.index = w_t.index.astype(str)
        w_t.index.name = None
        backtest_weights.loc[unique_dates[t]] = w_t
        backtest_weights = backtest_weights.fillna(0)

        if t%100 == 0:
            # Save results
            # Maybe use parquet
            backtest_weights.to_csv(f"{backtest_result_path}{save_filename}", index=True)

    backtest_weights.to_csv(f"{backtest_result_path}{save_filename}", index=True)

    return
    

if __name__ == "__main__":
    product_args = itertools.product(lambda_l1_list, lambda_l2_list, lambda_l3_list, top_n_assets_list)
    with Pool(4) as pool:
        results = pool.map(run_factor_port_backtest, product_args)
