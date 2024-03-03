import numpy as np
import pandas as pd
from scipy.linalg import qr, inv, eig
from scipy.linalg import sqrtm
from ipca import InstrumentedPCA
from . import pyport


def PRPCA_factor(X, T, N, K, gamma, stdnorm, t, Lambdahatprevious = None):
    """
    Apply PRPCA on X to get K principal components.

    Parameters
    ----------
    X : 2D numpy array
        The input data where each row is a data point, each column is a feature.

    T : int
        Time series length.

    N : int
        Number of features (variables).

    K : int
        Number of principal components to return.

    gamma : float
        Hyper-parameter used in the weighting matrix.

    stdnorm : int, optional
        Whether to standardize the input data. If 1, standardize. If 0, don't standardize.
        The default is 0.

    t : int
        Current time index.

    Lambdahatprevious : 2D numpy array
        The loading matrix (eigenvectors) from the previous time step.

    Returns
    -------
    Fhat : 2D numpy array
        The principal components (scores). Shape (N, K).

    Lambdahat : 2D numpy array
        The loadings (eigenvectors). Shape (N, K).

    Lambdahatprevious : 2D numpy array
        The updated loadings (eigenvectors). Shape (N, K).
    """
    if stdnorm == 1:
        WN = inv(sqrtm(np.diag(np.diag(X.T @ (np.eye(T) - np.ones((T, T)) / T) * X / T / N))))
    else:
        WN = np.eye(N)

    WT = (np.eye(T) + gamma * np.ones((T, T)) / T)

    # Generic estimator for general weighting matrices
    Xtilde = X @ WN

    # Covariance matrix with weighted mean
    VarWPCA = Xtilde.T @ WT @ Xtilde / N / T
    # Eigenvalue decomposition:  
    DWPCA, VWPCA = eig(VarWPCA)
    # DDWPCA has the eigenvalues
    DDWPCA = np.sort(DWPCA)[::-1]
    ID = np.argsort(DWPCA)[::-1]
    VWPCA = VWPCA[:, ID]
    # Lambdahat are the eigenvectors after reverting the cross-sectional transformation
    Lambdahat = inv(WN.T) @ VWPCA[:, :K]
    # Normalizing the signs of the loadings
    if t == 0:
        Lambdahat = Lambdahat @ np.diag(np.sign(np.mean(X @ Lambdahat @ inv(Lambdahat.T @ Lambdahat), axis=0)))
    else:
        Lambdahat = Lambdahat @ np.diag(np.sign(np.diag(Lambdahat.T @ Lambdahatprevious)))
    Lambdahatprevious = Lambdahat

    Fhat = X @ Lambdahat[:, :K] @ inv(Lambdahat[:, :K].T @ Lambdahat[:, :K])
    return Fhat, Lambdahat, Lambdahatprevious

def IPCA_factor(df, df_all, unique_dates, t, K, window_size):
    """
    Apply Instrumented PCA (IPCA) on a data frame to extract K principal components.

    Parameters
    ----------
    df : pandas.DataFrame
        The input data where each row is a data point, each column is a feature.

    unique_dates : list of datetime.date
        Unique dates present in the data.

    t : int
        Current time index.

    K : int
        Number of principal components to return.

    window_size : int
        Size of the rolling window for IPCA.
        
    Returns
    -------
    Gamma : 2D numpy array
        The loadings (coefficients) matrix from IPCA. Shape (N, K).

    Factors : pandas.DataFrame
        The principal components (scores) extracted from IPCA. 

    y_next : pandas.Series
        Returns for the next date after the current window.

    X_next : pandas.DataFrame
        Features for the next date after the current window.
    """
    # Select the date range for the current window
    selected_dates = unique_dates[t-window_size:t]
    # Identify the date immediately following the current window
    next_date = unique_dates[t]
    last_date = unique_dates[t-1]
    # Filter the data to include only rows within the selected date range
    windowed_data = df[df['date'].isin(selected_dates)]
    # Filter the data to include only the row for the next date
    next_data = df_all[df_all['date'] == next_date]
    
    # Remove columns in next_data that are all zeros in windowed_data
    next_data = next_data.loc[:, (windowed_data != 0).any(axis=0)]

    last_data = windowed_data[windowed_data['date'] == last_date]
    # Remove columns in next_data that are all zeros in windowed_data
    last_data = last_data.loc[:, (windowed_data != 0).any(axis=0)]

    # Remove columns in windowed_data that are all zeros
    windowed_data = windowed_data.loc[:, (windowed_data != 0).any(axis=0)]
    
    # Set 'cusip' and 'date' as indices for the windowed data
    windowed_data.set_index(['cusip', 'date'], inplace=True)
    next_data = next_data[next_data['cusip'].isin(last_data['cusip'].unique())]
    # Extract return column for the next date
    y_next = next_data['ret'] #lead return

    # Extract the cusip number for the last observation for a given window
    windowed_cusip = last_data['cusip'].unique()
    # Extract features (excluding 'cusip', 'permco', 'ret', 'date') for the next date
    X_next = next_data.drop(['cusip','permco', 'ret', 'date', 'prc'], axis=1)
    X_last = last_data.drop(['cusip','permco', 'ret', 'date', 'prc'], axis=1)

    # Extract return column for the windowed data
    y = windowed_data['ret'] #cur return

    # Extract features (excluding 'permco', 'ret') for the windowed data
    X = windowed_data.drop(['permco', 'ret', 'prc'], axis=1)

    # Create an instance of InstrumentedPCA with K factors and no intercept
    regr = InstrumentedPCA(n_factors=K, intercept=False, iter_tol=1e-4)

    # Fit the IPCA model on the windowed data
    regr = regr.fit(X=X, y=y, quiet = True)

    # Extract the loadings and factors from the IPCA model
    Gamma, Factors = regr.get_factors(label_ind=True)

    
    return Gamma, Factors, y_next, X_next, X_last, windowed_cusip

def PCA_factor(X, K, stdnorm, t, Lambdahatprevious = None):
    """
    Apply PCA on X to get K principal components.

    Parameters
    ----------
    X : 2D numpy array
        The input data where each row is a data point, each column is a feature.

    K : int
        Number of principal components to return.

    stdnorm : int, optional
        Whether to standardize the input data. If 1, standardize. If 0, don't standardize.
        The default is 1.

    Returns
    -------
    Fhat : 2D numpy array
        The principal components (scores). Shape (N, K).

    Lambdahat : 2D numpy array
        The loadings (eigenvectors). Shape (N, K).
    """

    # Standardize the data if stdnorm is 1
    if stdnorm == 1:
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    Cov = np.cov(X, rowvar=False)
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = eig(Cov)
    # Sort by eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]   
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    Lambdahat = eigenvectors[:,:K]

    if t == 0:
        Lambdahat = Lambdahat @ np.diag(np.sign(np.mean(X @ Lambdahat @ inv(Lambdahat.T @ Lambdahat), axis=0)))
    else:
        Lambdahat = Lambdahat @ np.diag(np.sign(np.diag(Lambdahat.T @ Lambdahatprevious)))
    Lambdahatprevious = Lambdahat

    Fhat = X @ Lambdahat[:, :K] @ inv(Lambdahat[:, :K].T @ Lambdahat[:, :K])

    return Fhat, Lambdahat, Lambdahatprevious


def IPCAOOS(df, df_all, stdnorm, gamma, K, window):
    """
    Conduct Out-of-sample evaluation of Regularized Principal Component Analysis (RP-PCA).

    Parameters
    ----------
    df : DataFrame
        Input Asness universe data in dataframe format, contains 'date' column.

    df : DataFrame
        Input all data in dataframe format, contains 'date' column.

    stdnorm : int
        Whether to standardize the input data. If 1, standardize. If 0, don't standardize.

    gamma : float
        The tuning parameter in RP-PCA which controls the trade-off between the loss function and the penalty term.

    K : int
        Number of principal components to return.

    window : int
        The size of the moving window used for the out-of-sample prediction.

    Returns
    -------
    SR_grid_ipca : list
        Sharpe ratios for different combinations of the tuning parameters for IPCA.

    maxreturntime_ipca : list
        Returns for different combinations of the tuning parameters for IPCA.
    """

    unique_dates = df['date'].unique()
    T = len(unique_dates)
    Sigmatime_ipca, mutime_ipca = [], []

    maxreturntime_ipca = [[None for _ in range(K)] for _ in range(100)] 
    # turnover:
    # df_prc = df.pivot(index='date', columns=['cusip'], values='prc')
    # # reorder the df_prc has the same columns order as df['cusip].unique()
    # df_prc = df_prc[df['cusip'].unique()]
    # df_prc = df_prc.reset_index(drop=True)
    # df_prc.columns.name = None
    w_df = pd.DataFrame(0, index=range(T),columns=df['cusip'].unique())
    w_adj = pd.DataFrame(0, index=range(T),columns=df['cusip'].unique())
    for t in range(window, T):
        print(t)
        maxreturntime_temp_ipca = [None] * (K-1) 
        k = t-window
        Gamma_all, factors_all, y_next_all, X_next_all, Lambdahat_IPCA_all, Sigmatime_ipca_all, mutime_ipca_all= [None] * (K-1), [None] * (K-1), [None] * (K-1), [None] * (K-1),[None] * (K-1) ,[None] * (K-1),[None] * (K-1) 
        for i in range(1,K):
            Gamma_temp, factors_temp, y_next_temp, X_next_temp, X_last_temp, windowed_cusip = IPCA_factor(df, df_all, unique_dates, t, i+1, window)
            Lambdahat_IPCA_temp = inv(Gamma_temp.T @ X_last_temp.T @ X_last_temp @ Gamma_temp) @ Gamma_temp.T @ X_last_temp.T
            Sigmatime_ipca_temp = np.cov(factors_temp, rowvar = True)
            mutime_ipca_temp = np.array(np.mean(factors_temp, axis=1))
            Gamma_all[i-1] = Gamma_temp
            factors_all[i-1] = factors_temp
            y_next_all[i-1] = y_next_temp
            X_next_all[i-1] = X_next_temp
            Lambdahat_IPCA_all[i-1] = Lambdahat_IPCA_temp
            Sigmatime_ipca_all[i-1] = Sigmatime_ipca_temp
            mutime_ipca_all[i-1] = mutime_ipca_temp

        g1 = np.exp(np.linspace( np.log(1e-6),np.log(5),10))
        g2 = np.exp(np.linspace( np.log(1e-6),np.log(5),10))
        # grid search
        for a in range(10):
                for b in range(10):
                    for i in range(1,K):
                        Gamma = Gamma_all[i-1]
                        y_next = y_next_all[i-1]
                        X_next = X_next_all[i-1]
                        Lambdahat_IPCA = Lambdahat_IPCA_all[i-1]
                        Sigmatime_ipca = Sigmatime_ipca_all[i-1]
                        mutime_ipca = mutime_ipca_all[i-1]
                        OptimalPortfolioWeightstime_temp_ipca = pyport.portfolio_optimization(meanVec = np.array(mutime_ipca),sigMat=np.array(Sigmatime_ipca),retTarget = 0,longShort = 0.2,maxAlloc=0.08,
                                                                        lambda_l1=g1[a],lambda_l2=g2[b],riskfree = 0,assetsOrder=None,maxShar = 1,factor=np.array(Lambdahat_IPCA.T)[:,0:i+1],turnover = None, 
                                                                        w_pre = None, individual = False, exposure_constrain = 0, w_bench = None, 
                                                                        factor_exposure_constrain = None, U_factor = None, general_linear_constrain = None, 
                                                                        U_genlinear = 0, w_general = None, TE_constrain = 0, general_quad = 0, Q_w = None, 
                                                                        Q_b = None, Q_bench = None)[0].reshape(-1,1)
                        ind = i-1
                        
                        if maxreturntime_temp_ipca[ind] is None:
                            maxreturntime_temp_ipca[ind] = (inv(Gamma.T @ X_next.T @ X_next @ Gamma) @ Gamma.T @ X_next.T @ y_next).T @ OptimalPortfolioWeightstime_temp_ipca
                        else:
                            maxreturntime_temp_ipca[ind] = np.vstack([maxreturntime_temp_ipca[ind], (inv(Gamma.T @ X_next.T @ X_next @ Gamma) @ Gamma.T @ X_next.T @ y_next).T @ OptimalPortfolioWeightstime_temp_ipca])

                        if maxreturntime_ipca[a * 10 + b][ind] is None:
                            maxreturntime_ipca[a * 10 + b][ind] = maxreturntime_temp_ipca[ind][a * 10 + b]
                        else:
                            maxreturntime_ipca[a * 10 + b][ind] = np.vstack([maxreturntime_ipca[a * 10 + b][ind], maxreturntime_temp_ipca[ind][a * 10 + b]])
                    
                    # turnover:
                    # if a == 2 and b == 3:
                    #     w_individual = np.dot(np.array(Lambdahat_IPCA.T), OptimalPortfolioWeightstime_temp_ipca).flatten()
                    #     w_df.loc[t-window, windowed_cusip] = w_individual
                    #     if t-window > 0:
                    #         # get the adjusted weight
                    #         arr = np.array(w_df.loc[t-window-1, windowed_cusip]*(df_prc.loc[t-window,windowed_cusip]/df_prc.loc[t-window-1,windowed_cusip]))
                    #         arr[np.isnan(arr) | np.isinf(arr)] = 0
                    #         w_adj.loc[t-window, windowed_cusip] = arr

    SR_grid_ipca = [[None for _ in range(K-1)] for _ in range(100)]
    for i in range(len(g1)*len(g2)):
        for j in range(K-1):
            maxreturnnormalizedtime_ipca = maxreturntime_ipca[i][j] / np.std(maxreturntime_ipca[i][j], axis=0)
            SR_grid_ipca[i][j]= np.mean(maxreturnnormalizedtime_ipca, axis=0)

    return SR_grid_ipca, maxreturntime_ipca, w_df, w_adj

if __name__ == "main":
    data = pd.read_csv("asness_withprc.csv")
    data = data.drop(['spread'], axis = 1)
    data = data.drop('Unnamed: 0', axis = 1)
    df = data.copy(deep = True)
    df_pre = data.loc[:, (data != 0).any(axis=0)]
    characteristics = df_pre.drop(['date','cusip','permco','prc'], axis = 1).columns
    df_all = pd.read_csv("df_all_withprc.csv")
    df_all = df_all.drop(['Unnamed: 0', 'spread'], axis=1)

    shar_ipca, ret_ipca, w_df, w_adj = IPCAOOS(df, df_all, stdnorm= 0, gamma = 15, K=6, window=240)



