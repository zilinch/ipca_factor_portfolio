import numpy as np
import pandas as pd
from scipy import sparse
import osqp
from numpy import linalg as la


def triple_sort(df, char1, char2, char3):
    """
    Sorts a given DataFrame based on three characteristics, divides each characteristic into quartiles,
    groups the data by date and quartiles, and calculates the market cap-weighted return for each group.
    
    Parameters
    ----------
    df: DataFrame
        Input DataFrame containing the data to be sorted.
    char1, char2, char3: str
        Names of the columns representing the three characteristics based on which the sorting is to be done.
    
    Returns
    -------
    portfolio_returns_pivot: DataFrame
        A DataFrame with dates as the index, quartile combinations as columns, and market cap-weighted returns as values.
    """

    # Create copy to avoid modifying original df
    df = df.copy()
    
    # Divide each characteristic into 4 quartiles
    df[char1+'_quartile'] = df.groupby('date')[char1].transform(lambda x: pd.qcut(x, 4, labels=False))
    df[char2+'_quartile'] = df.groupby('date')[char2].transform(lambda x: pd.qcut(x, 4, labels=False))
    df[char3+'_quartile'] = df.groupby('date')[char3].transform(lambda x: pd.qcut(x, 4, labels=False))
    df['triple_quantile'] = df['beta_quartile'].astype(str) + df['at_quartile'].astype(str) + df['ac_quartile'].astype(str)
    # Group by date and quartiles and calculate market cap-weighted return
    df = df.drop(columns=[char1+'_quartile', char2+'_quartile', char3+'_quartile'])
    portfolio_returns = df.groupby(['date', 'triple_quantile']).apply(
            lambda x: np.average(x['ret'], weights=x['mktcap'])).reset_index()
    portfolio_returns.columns = ['date', 'triple_quantile', 'ret']
    portfolio_returns_pivot = portfolio_returns.pivot(index='date', columns=['triple_quantile'], values='ret')
    return portfolio_returns_pivot

def single_sort(df, characteristics):
    """
    Sorts a given DataFrame based on a list of characteristics, divides each characteristic into deciles,
    and calculates the market cap-weighted return for each decile.
    
    Parameters
    ----------
    df: DataFrame
        Input DataFrame containing the data to be sorted.
    characteristics: list of str
        List of column names representing the characteristics based on which the sorting is to be done.
    
    Returns
    -------
    portfolio_returns: DataFrame
        A DataFrame containing the market cap-weighted return for each decile of each characteristic.
    """
    def weighted_return(group):
        return np.average(group['ret'], weights=group['mktcap'])

    # Initialize DataFrame to store portfolio returns
    portfolio_returns = pd.DataFrame(index=df['date'].unique())

    for char in characteristics:
        # Sort assets into deciles based on each characteristic
        df[char + '_decile'] = df.groupby('date')[char].transform(lambda x: pd.qcut(x, 10, labels=False, duplicates='drop'))
        
        char_portfolio_returns = df.groupby(['date', char + '_decile']).apply(weighted_return).reset_index()

        # Pivot the DataFrame from long format to wide format
        char_portfolio_returns = char_portfolio_returns.pivot(index='date', columns=char + '_decile', values=0)

        # Rename columns
        char_portfolio_returns.columns = [char + '_decile_' + str(i) for i in range(1, 11)]

        # Add the portfolio returns for this characteristic to the overall portfolio returns DataFrame
        portfolio_returns = pd.concat([portfolio_returns, char_portfolio_returns], axis=1)

    return portfolio_returns

def sigMatShrinkage(sigMat,lambda_l2, factor = None):
    """
    Applies shrinkage to the given covariance matrix using a specified shrinkage parameter and an optional factor matrix.
    
    Parameters
    ----------
    sigMat: array-like
        Input covariance matrix.
    lambda_l2: float
        Shrinkage parameter.
    factor: array-like, optional
        Factor matrix used for shrinkage, defaults to None.
    
    Returns
    -------
    sigMat: array-like
        Shrunken covariance matrix.
    """
    #import pdb; pdb.set_trace()
    d = sigMat.shape[0]
    sig = np.sqrt(np.diag(sigMat))
    t = np.dot(np.diag(sig**(-1)), sigMat)
    corrMat = np.dot(t, np.diag(sig**(-1)))
    corrs = None
    for k in range(d-1):
        if corrs is None:
            corrs = np.diag(corrMat, k+1)
        else:
            corrs = np.hstack([corrs, np.diag(corrMat,k+1)])
    if 1 == 1:
        if factor is not None:
            sigMat = sigMat + lambda_l2 * np.dot(factor.transpose(), factor)
        else:
            sigMat = sigMat + lambda_l2 * np.mean(sig**2)*np.eye(d)
    else:
        t = np.dot(np.mean(sig)*np.eye(d), np.eye(d)+(np.ones(d,d)-np.eye(d))* np.mean(corrs))
        sigMat = sigMat + lambda_l2 * np.dot(t, np.mean(sig)*np.eye(d))
    return sigMat

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    if isPD(A):
        return A
    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

def Dmat(n, k):
    """
    function reform a matrix for assets with order
    Parameters
    ----------
    n : int
    k : int

    Returns
    -------
    D : Array
    """
    if k == 0:
        D = np.eye(n)
    elif k == 1:
        D = np.eye(n - 1, n)
        for i in range(n - 1):
            D[i, i + 1] = -1
    else:
        D = Dmat(n, 1)
        for i in range(k - 1):
            Dn = Dmat(n - i - 1, 1)
            D = np.dot(Dn, D)
    return D
def constrain_matrix(d, N, meanvariance = 0, maxShar = 0, factor = None, meanVec = None, riskfree = 0, assetsOrder = None,  maxAlloc = 1, 
longShort = 0, lambda_l1 = 0, turnover = None, w_pre = None, individual = False, exposure_constrain = 0, 
w_bench = None, factor_exposure_constrain = None, U_factor = None, general_linear_constrain = None, U_genlinear = 0, w_general = None):

    """
    This function creates the constraint matrices for an optimization problem, given a set of parameters. 
    
    Parameters
    ----------
    d: int
        Number of assets
    meanvariance: int or float, optional
        Mean-variance constraint for the portfolio, defaults to 0
    maxShar: int or float, optional
        Maximum sharpe ratio constraint for the portfolio, defaults to 0
    meanVec: array-like, optional
        Mean return vector of assets, defaults to None
    riskfree: int or float, optional
        Risk free rate, defaults to 0
    assetsOrder: array-like, optional
        assets ordering constraints, defaults to None
    maxAlloc: int or float, optional
        Maximum allocation in a single asset, defaults to 1
    longShort: int or float, optional
        Long-Short constraint for the portfolio, defaults to 0
    lambda_l1: int or float, optional
        L1 regularization constraint, defaults to 0
    turnover: array or float, optional
        Turnover constraint, defaults to None
    w_pre: array-like, optional
        Initial weights, defaults to None
    exposure_constrain: int or float, optional
        Exposure constraint, defaults to 0
    w_bench: array-like, optional
        Weights of benchmark portfolio, defaults to None
    factor_exposure_constrain: array-like, optional
        Factor exposure constraints, defaults to None
    U_factor: array or float, optional
        Upper bound on factor exposure, defaults to None
    general_linear_constrain: array-like, optional
        General linear constraint, defaults to None
    U_genlinear: int or float, optional
        Upper bound on general linear constraint, defaults to 0
    w_general: array-like, optional
        Weights of general linear constraint, defaults to None
    
    Returns
    -------
    A, l, u: tuple of arrays
        Constraint matrices for the optimization problem
    """


    #factor_exposure_constrain 1xd vector for time t
    if longShort == 0:
        Aeq = np.ones(d) # sum(w) = 1
        Beq = 1
        LB = np.zeros(d) 
        UB = maxAlloc*np.ones(d)
        A = np.vstack([Aeq, np.eye(d)]) # 0 < w_i < maxAlloc
        l = np.hstack([Beq, LB])
        u = np.hstack([Beq, UB])
        if maxShar:
            if factor is not None:
                #factor here should be Nxk matrix
                A = np.vstack([np.hstack([factor, -np.eye(N), np.eye(N)]),
                               np.hstack([np.sum(factor,axis=0), np.zeros(2*N)]), # sum w'f = kappa
                               np.hstack([factor, np.zeros((N,2*N))]), # 0 < w_i*u_i <U
                               np.hstack([np.zeros(d), np.ones(N), np.zeros(N)]), # sum u_plus = kappa(1+longshort)
                               np.hstack([factor, np.zeros((N,2*N))]),
                               np.zeros(d+2*N),
                               np.hstack([meanVec, np.zeros(2*N)]),
                               np.hstack([np.zeros((2*N,d))], np.eye(2*N)),
                               np.hstack([np.zeros((2*N,d)), -1*np.eye(2*N)])])
                Bwuv = np.hstack([np.zeros(N), 1, maxAlloc*np.ones(N), 1+abs(longShort), np.zeros(N), 1, 0, Grenze*np.ones(2*N), np.zeros(2*N)])
                l = np.hstack([np.zeros(N+1), -np.inf*np.ones(N+1), np.zeros(N), -np.inf, 1, -np.inf*np.ones(4*N)])
                u = np.hstack([np.zeros(2*N+1), np.inf*np.ones(N), 0, -1e-12, 1, np.zeros(4*N)])
            else:
                A = np.vstack([A, # sum(w) = kappa
                -np.eye(d), # w_i >= 0
                np.zeros(d), # kappa > 0
                meanVec]) # kappa*w'mu = 1
                Bwuv = np.hstack([1, maxAlloc*UB, LB, 1, 0]) # [w kappa] where kappa>0 is a scalar for the rescaling
                l = np.hstack([0, -np.inf*np.ones(2*d+1), 1]) 
                u = np.hstack([np.zeros(2*d+1), -1e-12, 1])
        if assetsOrder:
            # ordering constraint w_1 >= w_2 >= w_3 >= ... >= w_d
            L_ine = np.hstack([-np.inf, -np.ones(d - 1)])
            A = np.vstack([A, -1 * Dmat(d, 1)])
            B = np.zeros(d-1)
            if maxShar:
                Bwuv = np.hstack([Bwuv, np.zeros(d-1)])
            
            l = np.hstack([l, -np.ones(d-1)])
            u = np.hstack([u, B])
        
        if meanvariance and meanVec.any():
            if riskfree:
                A = np.vstack([A, -meanVec+riskfree]) # w'mu + (1-w)rf > meanvariance
                l = np.hstack([np.zeros(d+1), -np.inf])
                u = np.hstack([u, -meanvariance+riskfree])
            else:
                A = np.vstack([A, -meanVec]) # w' * mu > meanvariance
                l = np.hstack([l, -np.inf])
                u = np.hstack([u, -meanvariance])
        
        if U_factor is not None and factor_exposure_constrain is not None:
            # factor_exposure_constrain can be a d x k matrix and U_factor can be a k length vector
            A = np.vstack([A, factor_exposure_constrain]) # abs(beta_k' * w) < U 
            l = np.hstack([l, -U_factor])
            u = np.hstack([u, U_factor])
            if maxShar:
                Bwuv = np.hstack([Bwuv, 0])
        
        if U_genlinear > 0 and general_linear_constrain is not None:
            # A_B (w − w_B ) ≤ u_B
            A = np.vstack([A, general_linear_constrain])
            l = np.hstack([l, 0])
            u = np.hstack([u, U_genlinear + sum(np.dot(general_linear_constrain, w_general.reshape(-1,1)))])
            if maxShar:
                Bwuv = np.hstack([Bwuv, 0])
                
        if turnover is not None and w_pre is not None:
            # expend to [w, w_p, w_n]
            # w_p: w_old-w(w-w_old > 0)
            # w_n: w-w_old(w-wold < 0)
            if individual == True:
                # turnover is a vector abs(w_old_i - w_i) < U_i
                A = np.hstack([A, np.zeros((len(A), 2*d))])
                A = np.vstack([A,
                    np.hstack([np.eye(d), np.eye(d), -np.eye(d)]), # w + w_p - w_n = w_old
                    np.hstack([np.zeros((d,d)), np.eye(d), np.eye(d)]), # abs(w_old-w) < turnover
                    np.hstack([np.zeros((d,d)), np.eye(d), np.zeros((d,d))]), # w_p_i < turnover
                    np.hstack([np.zeros((d,d)), np.zeros((d,d)), np.eye(d)]) # w_n_i < turnover
                    ])
                l = np.hstack([l, w_pre, np.zeros(d), np.zeros(2*d)])
                u = np.hstack([u, w_pre, turnover, turnover*np.ones(2*d)])
                if maxShar:
                    Bwuv = np.hstack([Bwuv, np.zeros(4*d)])
            else:
                # turnover is a float sum(w_old - w) < U
                A = np.hstack([A, np.zeros((len(A), 2*d))])
                A = np.vstack([A,
                    np.hstack([np.eye(d), np.eye(d), -np.eye(d)]),
                    np.hstack([np.zeros(d), np.ones(d), np.ones(d)]),
                    np.hstack([np.zeros((d,d)), np.eye(d), np.zeros((d,d))]),
                    np.hstack([np.zeros((d,d)), np.zeros((d,d)), np.eye(d)])
                    ])
                l = np.hstack([l, w_pre, 0, np.zeros(2*d)])
                u = np.hstack([u, w_pre, turnover, turnover*np.ones(2*d)])
                if maxShar:
                    Bwuv = np.hstack([Bwuv, np.zeros(3*d+1)])
        
        if exposure_constrain > 0 and w_bench is not None: 
            # sum(abs(w - w_bench)) < U
            A = np.hstack([A, np.zeros((len(A), 2*d))])
            A = np.vstack([A,
                np.hstack([np.eye(d), np.eye(d), -np.eye(d)]),
                np.hstack([np.zeros(d), np.ones(d), np.ones(d)]),
                np.hstack([np.zeros((d,d)), np.eye(d), np.zeros((d,d))]),
                np.hstack([np.zeros((d,d)), np.zeros((d,d)), np.eye(d)])
                ])
            l = np.hstack([l, w_bench, 0, np.zeros(2*d)])
            u = np.hstack([u, w_bench, exposure_constrain, exposure_constrain*np.ones(2*d)])
            if maxShar:
                Bwuv = np.hstack([Bwuv, np.zeros(3*d+1)])
    

    else:
        # the following two auxiliary variables are introduced for the long-short portfolio estimation
        # u = w.*(w>0) % postive part of w
        # v = -1*(w.*(w<0)) % negative part of w
        A = np.hstack([np.zeros(d), np.ones(d), np.zeros(d)])
        B = 1 + abs(longShort) # sum of u's <= 1+longShort
        Grenze = min(abs(longShort),maxAlloc) 
        Aeq = np.vstack([
            np.hstack([np.eye(d), -np.eye(d), np.eye(d)]), # w - u + v = 0
            np.hstack([np.ones(d), np.zeros(d), np.zeros(d)]) # sum(w) = 1
        ])
        Beq = np.hstack([np.zeros(d), 1]) 
        LB = np.hstack([-Grenze*np.ones(d), np.zeros(2*d)]) # w >= -Grenze
        UB = Grenze*np.ones(3*d) # [w,u,v] <= Grenze
        A = np.vstack([Aeq, A, np.eye(3*d)])
        l = np.hstack([Beq, 0, LB])
        u = np.hstack([Beq, B, UB])
        if maxShar:
            if factor is not None:
                #factor here should be Nxk matrix
                A = np.vstack([np.hstack([factor, -np.eye(N), np.eye(N)]), #w'f_i = u_plus - u_minus
                               np.hstack([np.sum(factor,axis=0), np.zeros(2*N)]), # sum w'f = kappa
                               np.hstack([factor, np.zeros((N,2*N))]), # -inf < w_i <U
                               np.hstack([np.zeros(d), np.ones(N), np.zeros(N)]), # sum u_plus = kappa(1+longshort)
                               np.hstack([factor, np.zeros((N,2*N))]), # L < w_i < inf
                               np.zeros(d+2*N), # kappa > 0
                               np.hstack([meanVec, np.zeros(2*N)]),
                               np.hstack([np.zeros((2*N,d)), np.eye(2*N)]),
                               np.hstack([np.zeros((2*N,d)), -1*np.eye(2*N)])])
                
                Bwuv = np.hstack([np.zeros(N), 1, Grenze*np.ones(N), 1+abs(longShort), -Grenze*np.ones(N), 1, 0, Grenze*np.ones(2*N), np.zeros(2*N)])
                l = np.hstack([np.zeros(N+1), -np.inf*np.ones(N), -np.inf, np.zeros(N), -np.inf, 1, -np.inf*np.ones(4*N)])
                u = np.hstack([np.zeros(N+1), np.zeros(N), 0, np.inf*np.ones(N), -1e-12, 1, np.zeros(4*N)])
            else:
                A = np.vstack([A, -np.eye(3*d), np.zeros(3*d), 
                np.hstack([meanVec, np.zeros(2*d)])])
                Bwuv = np.hstack([np.zeros(d), 1, (1+abs(longShort)), UB, -LB, 1, 0]) # [w u v kappa] where kappa>0 is a scalar for the rescaling
                l = np.hstack([np.zeros(d+1), -np.inf*np.ones(6*d+2), 1])
                u = np.hstack([np.zeros(7*d+2), -1e-12, 1])

        if assetsOrder:
            A = np.vstack([A,
                np.hstack([-1*Dmat(d,1), np.zeros((d-1, 2*d))])
            ])
            if maxShar:
                Bwuv = np.hstack([Bwuv, np.zeros(d-1)])
            l = np.hstack([l, -(1+2*Grenze)*np.ones(d-1)])
            u = np.hstack([u, np.zeros(d-1)])
        
        if meanvariance and meanVec.any():
            if riskfree:
                A = np.vstack([A, np.hstack([-meanVec+riskfree, np.zeros(2 * d)])]) # w'mu + (1-w)rf > meanvariance
                l = np.hstack([np.zeros(d+1), 0, LB, -np.inf])
                u = np.hstack([u, -meanvariance+riskfree])
            else:
                A = np.vstack([A, np.hstack([-meanVec, np.zeros(2 * d)])]) # w' * mu > meanvariance
                l = np.hstack([l, -np.inf])
                u = np.hstack([u, -meanvariance])
        
        if U_factor is not None and factor_exposure_constrain is not None:
            # factor_exposure_constrain can be a d x k matrix and U_factor can be a k length vector
            A = np.vstack([A, 
            np.hstack([factor_exposure_constrain, np.zeros(2*d)])]) # abs(beta_k' * w) < U 
            l = np.hstack([l, -U_factor])
            u = np.hstack([u, U_factor])
            if maxShar:
                Bwuv = np.hstack([Bwuv, 0])
        if U_genlinear > 0 and general_linear_constrain is not None:
            # A_B (w − w_B ) ≤ u_B
            A = np.vstack([A, 
            np.hstack([general_linear_constrain, np.zeros(2*d)])])
            l = np.hstack([l, 0])
            u = np.hstack([u, U_genlinear + sum(np.dot(general_linear_constrain, w_general.reshape(-1,1)))])
            if maxShar:
                Bwuv = np.hstack([Bwuv, 0])

        if turnover is not None and w_pre is not None:
            # expend to [w, w_p, w_n]
            # w_p: w_old-w(w-w_old > 0)
            # w_n: w-w_old(w-wold < 0)
            if individual == True:
                A = np.hstack([A, np.zeros((len(A), 2*d))])
                A = np.vstack([A,
                    np.hstack([np.eye(d), np.zeros((d,d)), np.zeros((d,d)), np.eye(d), -np.eye(d)]),
                    np.hstack([np.zeros((d,d)), np.zeros((d,d)), np.zeros((d,d)), np.eye(d), np.eye(d)]),
                    np.hstack([np.zeros((d,d)), np.zeros((d,d)), np.zeros((d,d)), np.eye(d), np.zeros((d,d))]),
                    np.hstack([np.zeros((d,d)), np.zeros((d,d)), np.zeros((d,d)), np.zeros((d,d)), np.eye(d)])
                    ])
                l = np.hstack([l, w_pre, np.zeros(3*d)])
                u = np.hstack([u, w_pre, turnover, np.hstack([turnover, turnover])*np.ones(2*d)])
                if maxShar:
                    Bwuv = np.hstack([Bwuv, np.zeros(4*d)])
            else:
                A = np.hstack([A, np.zeros((len(A), 2*d))])
                A = np.vstack([A,
                    np.hstack([np.eye(d), np.zeros((d,d)), np.zeros((d,d)), np.eye(d), -np.eye(d)]),
                    np.hstack([np.zeros(d), np.zeros(d), np.zeros(d), np.ones(d), np.ones(d)]),
                    np.hstack([np.zeros((d,d)), np.zeros((d,d)), np.zeros((d,d)), np.eye(d), np.zeros((d,d))]),
                    np.hstack([np.zeros((d,d)), np.zeros((d,d)), np.zeros((d,d)), np.zeros((d,d)), np.eye(d)])
                    ])
                l = np.hstack([l, w_pre, 0, np.zeros(2*d)])
                u = np.hstack([u, w_pre, turnover, turnover*np.ones(2*d)])
                if maxShar:
                    Bwuv = np.hstack([Bwuv, np.zeros(3*d+1)])
        
        if exposure_constrain > 0 and w_bench is not None:
            # sum(abs(w - w_bench)) < U
            A = np.hstack([A, np.zeros((len(A), 2*d))])
            A = np.vstack([A,
                np.hstack([np.eye(d), np.zeros((d,d)), np.zeros((d,d)), np.eye(d), -np.eye(d)]),
                np.hstack([np.zeros(d), np.zeros(d), np.zeros(d), np.ones(d), np.ones(d)]),
                np.hstack([np.zeros((d,d)), np.zeros((d,d)), np.zeros((d,d)), np.eye(d), np.zeros((d,d))]),
                np.hstack([np.zeros((d,d)), np.zeros((d,d)), np.zeros((d,d)), np.zeros((d,d)), np.eye(d)])
                ])
            l = np.hstack([l, w_bench, 0, np.zeros(2*d)])
            u = np.hstack([u, w_bench, exposure_constrain, exposure_constrain*np.ones(2*d)])
            if maxShar:
                Bwuv = np.hstack([Bwuv, np.zeros(3*d+1)])
    
    if maxShar:
        # add kappa to constrain matrix
        A = np.hstack([A, -Bwuv.reshape(-1,1)])
        
    return A, l, u
        
def penalty_vector(d, N, sigMat, maxShar = 0, factor = None, longShort=0, lambda_l1=0, turnover = None, exposure_constrain = 0, TE_constrain = None, Q_b = None, Q_bench = None):
    """
    This function calculates the penalty vector for an optimization problem, given a set of parameters. 
    
    Parameters
    ----------
    d: int
        Number of assets
    sigMat: array-like
        Covariance matrix of assets
    longShort: int or float, optional
        Long-Short constraint for the portfolio, defaults to 0
    lambda_l1: int or float, optional
        L1 regularization term, defaults to 0
    factor: array, optional
        Eigen value of factor model
    turnover: array or float, optional
        Turnover constraint, defaults to None
    exposure_constrain: int or float, optional
        Exposure constraint, defaults to 0
    TE_constrain: array-like, optional
        Tracking error constraint, defaults to None
    Q_b: array-like, optional
        Quadratic term for bias, defaults to None
    Q_bench: array-like, optional
        Benchmark for quadratic term, defaults to None
    
    Returns
    -------
    meanVec: array
        Penalty vector
    """
    if longShort == 0:
        if lambda_l1:
            # lambda_l1 * w
            if factor is not None:
                meanVec = np.hstack([np.zeros(d), -lambda_l1 * np.ones(2 * N)])
            else:
                meanVec = -lambda_l1*np.ones(d)
        else:
            meanVec = -np.zeros(d)
        if TE_constrain:
            # (w − wB )′Σ(w − wB )
            meanVec = meanVec + 2*np.dot(TE_constrain, sigMat)
        if turnover is not None or exposure_constrain:
            # expend the weight vector
            meanVec = np.hstack([meanVec, np.zeros(2*d)])
        if Q_b:
            meanVec = meanVec + 2*np.dot(Q_bench, Q_b)
    else:
        if lambda_l1:
            # lambda_l1 * (u+v)
            if factor is not None:
                meanVec = np.hstack([np.zeros(d), -lambda_l1 * np.ones(2 * N)])
            else:
                meanVec = np.hstack([np.zeros(d), -lambda_l1 * np.ones(2 * d)])
        else:
            meanVec = np.hstack([np.zeros(d), np.zeros(2 * d)])
        if TE_constrain:
            meanVec = meanVec + np.hstack([np.zeros(d), np.dot(TE_constrain, sigMat), -np.dot(TE_constrain, sigMat)])
        if turnover is not None or exposure_constrain:
            meanVec = np.hstack([meanVec, np.zeros(2*d)])
        if Q_b:
            meanVec = meanVec + np.hstack([np.zeros(d), np.dot(Q_bench, Q_b), -np.dot(Q_bench, Q_b)])
    if maxShar:
        meanVec = np.hstack([meanVec, 0])
    return meanVec
def sigMat_expend(d, N, sigMat, maxShar = 0, factor = None, longShort = 0, turnover = None, exposure_constrain = 0, TE_constrain = 0, general_quad = 0, Q_w = None, Q_b = None):
    """
    This function expands the covariance matrix for an optimization problem, given a set of parameters. 
    
    Parameters
    ----------
    d: int
        Number of assets
    sigMat: array-like
        Covariance matrix of assets
    longShort: int or float, optional
        Long-Short constraint for the portfolio, defaults to 0
    turnover: array or float, optional
        Turnover constraint, defaults to None
    exposure_constrain: int or float, optional
        Exposure constraint, defaults to 0
    TE_constrain: int or float, optional
        Tracking error constraint, defaults to 0
    general_quad: int or float, optional
        General quadratic constraint, defaults to 0
    Q_w: array-like, optional
        Quadratic term for weights, defaults to None
    Q_b: array-like, optional
        Quadratic term for bias, defaults to None
    
    Returns
    -------
    sigMat: array
        Expanded covariance matrix
    """
    if Q_w:
        sigMat += Q_w
    if Q_b:
        sigMat += Q_b
    if factor is not None:
        sigMat = np.vstack([np.hstack([sigMat, np.zeros((d, 2*N))]),
                            np.zeros((2*N, 2*N+d))])
    else:
        if longShort == 0:
            if turnover is not None or exposure_constrain:
                sigMat = np.vstack([
                    np.hstack([sigMat, np.zeros((d,2*d))]),
                    np.zeros((2*d,3*d))
                ])
            if TE_constrain or general_quad:
                sigMat *= 2
        else:
            sigMat = np.vstack([
                np.hstack([sigMat, np.zeros((d,2*d))]),
                np.zeros((2*d,3*d))
            ])
            if turnover is not None or exposure_constrain:
                sigMat = np.vstack([
                np.hstack([sigMat, np.zeros((3*d,2*d))]),
                np.zeros((2*d,5*d))
                ])
            if TE_constrain or general_quad:
                sigMat *= 2
    if maxShar:
        # add kappa
        sigMat = np.vstack([
            np.hstack([sigMat, np.zeros((sigMat.shape[0],1))]),
            np.zeros(sigMat.shape[0]+1)
        ])
    return sigMat

def portfolio_optimization(meanVec,sigMat,retTarget,longShort,maxAlloc=1,lambda_l1=0,lambda_l2=0,riskfree = 0,assetsOrder=None,maxShar = 0, factor = None,
        turnover = None, w_pre = None, individual = False, exposure_constrain = 0, w_bench = None, factor_exposure_constrain = None, U_factor = None, 
        general_linear_constrain = None, U_genlinear = 0, w_general = None, TE_constrain = 0, general_quad = 0, Q_w = None, Q_b = None, Q_bench = None
):
    """
    function do the portfolio optimization

    Parameters
    ----------
    retTarget : Float
        Target returns in percentage for optimizer. Takes values between 0 and 100
    LongShort : Float
        Takes value between 0 and 1
    sigMat: array-like
        Covariance matrix of assets
    maxAlloc : Float
        Takes value between 0 and 1. Specifies the maximum weight an asset can get
    lambda_l1 : Float
        Takes a value greater than 0. Specifies L1 penalty
    lambda_l2 : Float
        Takes a value greater than 0. Specifies L2 penalty
    maxShar: int or float, optional
        Maximum sharpe ratio constraint for the portfolio, defaults to 0
    factor: array-like
        Nxk matrix
    meanVec: array-like, optional
        Mean return vector of assets, defaults to None
    riskfree: int or float, optional
        Risk free rate, defaults to 0
    assetsOrder: array-like, optional
        assets ordering constraints, defaults to None
    individual : bool
        Individual turnover constrain, defaults to None
    turnover: array or float, optional
        Turnover constraint, defaults to None
    w_pre: array-like, optional
        Initial weights, defaults to None
    exposure_constrain: int or float, optional
        Exposure constraint, defaults to 0
    w_bench: array-like, optional
        Weights of benchmark portfolio, defaults to None
    factor_exposure_constrain: array-like, optional
        Factor exposure constraints, defaults to None
    U_factor: array or float, optional
        Upper bound on factor exposure, defaults to None
    general_linear_constrain: array-like, optional
        General linear constraint, defaults to None
    U_genlinear: int or float, optional
        Upper bound on general linear constraint, defaults to 0
    w_general: array-like, optional
        Weights of general linear constraint, defaults to None
    TE_constrain: int or float, optional
        Tracking error constraint, defaults to 0
    general_quad: int or float, optional
        General quadratic constraint, defaults to 0
    Q_w: array-like, optional
        Quadratic term for weights, defaults to None
    Q_b: array-like, optional
        Quadratic term for bias, defaults to None
    Q_bench: array-like, optional
        Benchmark for quadratic term, defaults to None

    Returns
    -------
    w_opt : Array
        Returns the weights of given to each asset in form of a numpy array
    var_opt : Float
        Returns the variance of the portfolio
    """
    if retTarget:
        dailyRetTarget = retTarget
        minEret = min(meanVec)
        maxEret = max(meanVec)
        if (dailyRetTarget < minEret) or (maxEret < dailyRetTarget):
            part1 = minEret
            part2 = min(maxEret, dailyRetTarget)
            dailyRetTarget = max(part1, part2)

    d = sigMat.shape[0]
    if assetsOrder:
        temp = sigMat[:, assetsOrder]
        sigMat = temp[assetsOrder, :]
        if retTarget or maxShar:
            meanVec = meanVec[assetsOrder]
    if lambda_l2:
        sigMat_shrik = sigMatShrinkage(sigMat, lambda_l2, factor)
        sigMat_shrik = nearestPD(sigMat_shrik)
    else:
        sigMat_shrik = nearestPD(sigMat)
    # import pdb; pdb.set_trace()
    if factor is not None:
        N = factor.shape[0]
    else:
        N = 0
    A, l, u = constrain_matrix(d, N, retTarget, maxShar, factor, meanVec, riskfree, assetsOrder,  maxAlloc, longShort, lambda_l1, 
    turnover, w_pre, individual, exposure_constrain, w_bench, factor_exposure_constrain, U_factor, general_linear_constrain, U_genlinear, w_general)
    sigMat_exp = sigMat_expend(d, N, sigMat_shrik, maxShar, factor, longShort, turnover, exposure_constrain, TE_constrain, general_quad, Q_w, Q_b)
    meanVec = penalty_vector(d, N, sigMat_shrik, maxShar, factor, longShort, lambda_l1, turnover, exposure_constrain, TE_constrain, Q_b, Q_bench)

    P = sparse.csc_matrix(sigMat_exp)
    A = sparse.csc_matrix(A)
    prob = osqp.OSQP()
    # Setup workspace
    prob.setup(P, -meanVec, A, l, u, verbose=False, max_iter=10000, eps_abs=1e-8, eps_rel=1e-8, eps_prim_inf=1e-8,
                eps_dual_inf=1e-8)
    # Solve problem
    res = prob.solve()
    w_opt = res.x
    
    if maxShar:
        if not w_opt.all():
            w_opt = np.ones(d) / d
        else:
            w_opt = w_opt[:d]/w_opt[-1]
    else:
        if not w_opt.all():
            w_opt = np.ones(d) / d
        else:
            w_opt = w_opt[:d]
            
    Var_opt = np.dot(np.dot(w_opt, sigMat), w_opt.transpose())
    if assetsOrder:
        w_opt = w_opt[assetsOrder]
    
    return w_opt, Var_opt