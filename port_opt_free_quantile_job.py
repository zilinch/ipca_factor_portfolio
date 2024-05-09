import numpy as np
import pandas as pd
from scipy.linalg import inv
import sys
import os
from multiprocessing import Pool
import itertools

from factor_port_opt import portfolio_optimization_cvx_tc_tangency

WINDOW_SIZE = 240 # Just decides which index to start
TARGET_EXPOSURE = 2

BACKTEST_RESULT_PATH = f"results/backtest_weights/ipca_v8/free_cvxopt/"
if not os.path.exists(BACKTEST_RESULT_PATH):
    os.makedirs(BACKTEST_RESULT_PATH)
IPCA_INTERMEDIATE_PATH = f"results/IPCA_intermediates_v8/"

LAMBDA_L1_LIST = [0]
LAMBDA_L2_LIST = [0]
LAMBDA_L3_LIST = [100]
TOP_N_ASSETS_LIST = [1000]
FACTOR_MEAN_METHOD_LIST = ["kalman"]

N_PROCESSES = 4

def run_factor_port_backtest(args):
    # Unpack arguments
    lambda_l1, lambda_l2, lambda_l3, n_assets_to_trade, factor_mean_method = args
    n_assets_to_trade = int(n_assets_to_trade)
    
    save_filename = f"{n_assets_to_trade}_assets_free_{lambda_l1}_{lambda_l2}_{lambda_l3}_{factor_mean_method}.csv"
    
    # load raw data to prepare costs
    # df_ipca is only used to get index and prepare TC, port optimization reads IPCA intermediates
    df_ipca = pd.read_pickle("data/kelly_data_without_nanocap.p")
    df_ipca = df_ipca[["id","eom","ret_local","ret_exc","prc"]]
    # The pickle file's "eom" column is already in datetime.date format, this is just in case
    df_ipca.eom = pd.to_datetime(df_ipca.eom).dt.date
    df_ipca.sort_values(by='eom', inplace=True, ignore_index=True)

    unique_dates = sorted(df_ipca['eom'].unique()) # unique dates
    T = len(unique_dates)
    test_period = len(unique_dates) - WINDOW_SIZE

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
    if os.path.isfile(f"{BACKTEST_RESULT_PATH}{save_filename}"):
        backtest_weights = pd.read_csv(f"{BACKTEST_RESULT_PATH}{save_filename}", index_col=0)
        backtest_weights.index = pd.to_datetime(backtest_weights.index).date
        # Find the index of the first row where sum of absolute weights is 0
        continue_date = backtest_weights.loc[(abs(backtest_weights).sum(axis=1) == 0)].index[0]
        df_ipca = df_ipca.query("eom >= @continue_date").reset_index(drop=True)
        df_ipca.sort_values(by='eom', inplace=True, ignore_index=True)
        #
        num_dates_left = len(df_ipca.eom.unique())
        t_shift = test_period - num_dates_left
        print(f"Skipped {t_shift} dates, starting at {unique_dates[WINDOW_SIZE-1 + t_shift]}")
    else:
        t_shift = 0
        backtest_weights = pd.DataFrame(columns=sorted(df_ipca.id.unique().astype(str)),
                                    index=unique_dates[WINDOW_SIZE-1:WINDOW_SIZE+test_period])

    backtest_weights.columns = backtest_weights.columns.astype(str)
    backtest_weights = backtest_weights.fillna(0)
    
    del df_ipca

    for t in range(WINDOW_SIZE-1 + t_shift, len(unique_dates)-1):
        # t+1 is date to predict, t is current date, t-1 is previous date
        if t%20 == 0:
            print(f"========Progress for L1={lambda_l1}, L2={lambda_l2}, L3={lambda_l3}========")
            print(f"{t} / {T} completed, now at {str(unique_dates[t])}")
        
        date_to_predict, Gamma, Factors, r_t, excess_r_t, X_last \
            = pd.read_pickle(f'{IPCA_INTERMEDIATE_PATH}predicting_{unique_dates[t+1]}.pickle')
        X_last.index = X_last.index.astype(str)

        # Choose top n assets with largest market cap (can implement a more rigourous version in the future)
        index_to_trade = X_last.market_equity.nlargest(n_assets_to_trade).index.astype(str)
        index_to_trade = index_to_trade.sort_values()
        X_last = X_last.loc[index_to_trade]
        
        # Get previous weights (updated to this month)
        if t == WINDOW_SIZE-1:
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
        try:
            V_t = inv(Gamma.T @ X_last.T @ X_last @ Gamma) @ Gamma.T @ X_last.T
        except:
            # Skip if can't find inverse
            continue

        # Kalman filter for predicting factor mean -------------------------------------
        def kalman_filter(z, x_est, P, A, H, Q, R, B=None, u=None):
            """
            Performs one step of the Kalman filter process for multi-dimensional data.

            Parameters:
            z (np.array): The measurement vector.
            x_est (np.array): The initial state estimate.
            P (np.array): The initial estimate covariance matrix.
            A (np.array): The state transition matrix.
            H (np.array): The observation matrix.
            Q (np.array): The process noise covariance matrix.
            R (np.array): The measurement noise covariance matrix.
            B (np.array, optional): The control input matrix.
            u (np.array, optional): The control input vector.

            Returns:
            np.array: The updated state estimate.
            np.array: The updated estimate covariance matrix.
            """
            # Prediction step
            if B is not None and u is not None:
                x_pred = A @ x_est + B @ u
            else:
                x_pred = A @ x_est

            P_pred = A @ P @ A.T + Q

            # Measurement update step
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            x_est = x_pred + K @ (z - H @ x_pred)
            P = (np.eye(len(x_est)) - K @ H) @ P_pred

            return x_est, P

        dim = Gamma.shape[1]

        x_est = np.zeros(dim)  # Initial state estimate
        P = np.eye(dim)  # Initial covariance matrix
        A = np.eye(dim)  # State transition matrix
        H = np.eye(dim)  # Observation matrix
        q_init = 0.01
        Q = q_init * np.eye(dim)  # Process noise covariance
        R = np.eye(dim)  # Measurement noise covariance

        measurements = Factors.T.cumsum().to_numpy()

        x_est_list = []

        phi = 0.1
        count = 0
        for z in measurements:
            x_est, P = kalman_filter(z, x_est, P, A, H, Q, R)

            std = np.sqrt(H @ P @ H.T + R)
            std = std[0][0]

            mae = np.mean(np.abs(x_est-z))
            if abs(mae) > 3*std:
                count += 1
                Q = (q_init+phi*count) * np.eye(dim)
            elif count > 0:
                count -= 1
                Q = (q_init+phi*count) * np.eye(dim)

            x_est_list.append(x_est)
        # ------------------------------------------------------------------------------

        Sigma_t = np.cov(Factors, rowvar = True)
        # Different factor means
        if factor_mean_method == "mean":
            mu_t = np.array(np.mean(Factors, axis=1))
        elif factor_mean_method == "kalman":
            mu_t = np.array(x_est_list[-1] - x_est_list[-2])

        optimal_w_asset = portfolio_optimization_cvx_tc_tangency(
            meanVec   = mu_t,
            sigMat    = Sigma_t,
            V         = V_t.T,
            targetExposure=TARGET_EXPOSURE,
            w_prev    = w_prev,
            costVec   = costVec,
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
            backtest_weights.to_csv(f"{BACKTEST_RESULT_PATH}{save_filename}", index=True)

    backtest_weights.to_csv(f"{BACKTEST_RESULT_PATH}{save_filename}", index=True)

    return
    

if __name__ == "__main__":
    product_args = itertools.product(LAMBDA_L1_LIST, LAMBDA_L2_LIST, LAMBDA_L3_LIST, TOP_N_ASSETS_LIST, FACTOR_MEAN_METHOD_LIST)
    with Pool(N_PROCESSES) as pool:
        results = pool.map(run_factor_port_backtest, product_args)
