import pandas as pd
import numpy as np
from scipy import sparse
import osqp
from cvxopt import solvers
from cvxopt import matrix

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

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

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
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
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
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def portfolio_optimization_tc(
        meanVec,sigMat,V,
        longShort,maxAlloc=1,
        lambda_l1=0,lambda_l2=0,lambda_l3=0,
        w_prev = None, costVec = None
):
    """
    function do the portfolio optimization

    Parameters
    ----------
    sigMat: array-like
        Covariance matrix of assets
    LongShort : Float
        Takes value between 0 and 1, maximum total short position allowed
    maxAlloc : Float
        Takes value between 0 and 1. Specifies the maximum weight an asset can get
    lambda_l1 : Float
        Takes a value greater than 0. Specifies L1 penalty
    lambda_l2 : Float
        Takes a value greater than 0. Specifies L2 penalty
    lambda_l3 : Float
        Takes a value greater than 0. Specifies L3 penalty
    V: array-like
        Nxk matrix, corresponds to the V matrix in the paper
    meanVec: array-like, optional
        Mean return vector of assets, defaults to None
    w_prev: array-like, optional
        Previous weights of assets, default to 0

    Returns
    -------
    w_opt : Array
        Returns the weights of given to each asset in form of a numpy array
    var_opt : Float
        Returns the variance of the portfolio
    """
    # Process input
    meanVec = np.array(meanVec).flatten()
    sigMat = np.array(sigMat)
    V = np.array(V)
    w_prev = np.array(w_prev).flatten()
    costVec = np.array(costVec)

    Grenze = min(abs(longShort),maxAlloc)
    d = sigMat.shape[0]
    N = V.shape[0]

    sigMat = sigMatShrinkage(sigMat, lambda_l2, V)
    sigMat = nearestPD(sigMat)

    # factor here should be Nxk matrix, corresponds to V in the paper
    #                         w_tou                    v_plus, v_minus            v_tou_plus, v_tou_minus
    A = np.vstack([np.hstack([V,                       -np.eye(N), np.eye(N),     np.zeros((N,2*N))]), #w'f_i = u_plus - u_minus
                   np.hstack([np.sum(V,axis=0),        np.zeros(2*N),             np.zeros(2*N)]), # sum w'f = kappa
                   np.hstack([V,                       np.zeros((N,2*N)),         np.zeros((N,2*N))]), # -inf < w_i <U
                   np.hstack([np.zeros(d),             np.ones(N), np.zeros(N),   np.zeros(2*N)]), # sum u_plus = kappa(1+longshort)
                   np.hstack([V,                       np.zeros((N,2*N)),         np.zeros((N,2*N))]), # L < w_i < inf
                   np.zeros(  d+                       2*N+                       2*N), # kappa > 0
                   np.hstack([meanVec,                 np.zeros(2*N),             np.zeros(2*N)]),
                   np.hstack([np.zeros((2*N,d)),       np.eye(2*N),               np.zeros((2*N,2*N))]),
                   np.hstack([np.zeros((2*N,d)),       -1*np.eye(2*N),            np.zeros((2*N,2*N))]),
                   np.hstack([-V,                      np.zeros((N,2*N)),         np.eye(N), -1*np.eye(N)]), # V_tou - V w_tou + kappa w_asset_prev = 0
                   np.hstack([np.zeros((2*N,d)),       np.zeros((2*N,2*N)),       np.eye(2*N)]),
                   np.hstack([np.zeros((2*N,d)),       np.zeros((2*N,2*N)),       -1*np.eye(2*N)])])

    # Change the second thing to 0 to get 0 sum weights, need to normalize at the end though
    Bwuv = np.hstack([np.zeros(N), 1, Grenze*np.ones(N), 1+abs(longShort), -Grenze*np.ones(N), 1, 0, Grenze*np.ones(2*N), np.zeros(2*N), -w_prev, Grenze*np.ones(2*N), np.zeros(2*N)])
    l = np.hstack([np.zeros(N+1), -np.inf*np.ones(N), -np.inf, np.zeros(N), -np.inf, 1, -np.inf*np.ones(4*N), np.zeros(N), -np.inf*np.ones(4*N)])
    u = np.hstack([np.zeros(N+1), np.zeros(N), 0, np.inf*np.ones(N), -1e-12, 1, np.zeros(4*N), np.zeros(N), np.zeros(4*N)])

    # add kappa to constrain matrix
    A = np.hstack([A, -Bwuv.reshape(-1,1)])

    # Expand sigma matrix, this is a square matrix
    target_dim = A.shape[1]
    sigMat_exp = np.vstack([
        np.hstack([sigMat, np.zeros((sigMat.shape[0],target_dim-sigMat.shape[1]))]),
        np.zeros((target_dim-sigMat.shape[0], target_dim))
    ])

    q = np.hstack([np.zeros(d), lambda_l1 * np.ones(2 * N), lambda_l3*costVec, lambda_l3*costVec, 0])

    P = sparse.csc_matrix(sigMat_exp)
    A = sparse.csc_matrix(A)
    prob = osqp.OSQP()
    # Setup QP problem
    prob.setup(P, q, A, l, u, verbose=False, max_iter=10000, eps_abs=1e-4, eps_rel=1e-4, eps_prim_inf=1e-4,
                eps_dual_inf=1e-4)
    # Solve problem
    res = prob.solve()

    optimal_asset_weights = (res.x[d:d+N] - res.x[d+N:d+2*N]) / res.x[-1]
    
    return optimal_asset_weights

def portfolio_optimization_cvx_tc(
        meanVec,sigMat,V,
        longShort,maxAlloc=1,
        lambda_l1=0,lambda_l2=0,lambda_l3=0,
        w_prev = None, costVec = None
):
    """
    function do the portfolio optimization

    Parameters
    ----------
    sigMat: array-like
        Covariance matrix of assets
    LongShort : Float
        Takes value between 0 and 1, maximum total short position allowed
    maxAlloc : Float
        Takes value between 0 and 1. Specifies the maximum weight an asset can get
    lambda_l1 : Float
        Takes a value greater than 0. Specifies L1 penalty
    lambda_l2 : Float
        Takes a value greater than 0. Specifies L2 penalty
    lambda_l3 : Float
        Takes a value greater than 0. Specifies L3 penalty
    V: array-like
        Nxk matrix, corresponds to the V matrix in the paper
    meanVec: array-like, optional
        Mean return vector of assets, defaults to None
    w_prev: array-like, optional
        Previous weights of assets, default to 0

    Returns
    -------
    w_opt : Array
        Returns the weights of given to each asset in form of a numpy array
    var_opt : Float
        Returns the variance of the portfolio
    """
    # Process input
    meanVec = np.array(meanVec).flatten()
    sigMat = np.array(sigMat)
    V = np.array(V)
    w_prev = np.array(w_prev).flatten().reshape(-1, 1) # column matrix
    costVec = np.array(costVec)

    d = sigMat.shape[0]
    N = V.shape[0]

    sigMat = sigMatShrinkage(sigMat, lambda_l2, V)
    sigMat = nearestPD(sigMat)

    # G x <= h
    # A x = b
    # V here should be Nxk matrix, corresponds to V in the paper
    #                         w_tou                 v_plus, v_minus             v_tou_plus, v_tou_minus    k
    G = np.vstack([np.hstack([np.zeros(d),          np.ones(N), np.zeros(N),    np.zeros(2*N),             -(1+longShort)]),             # sum v_plus / kappa <= (1+LongShort)
                   np.hstack([np.zeros(d),          np.zeros(N), np.ones(N),    np.zeros(2*N),             -longShort]),                 # sum v_minus / kappa <= LongShort
                   np.hstack([np.zeros(d),          np.zeros(2*N),              np.zeros(2*N),             -1]),                         # kappa > 0
                   np.hstack([np.zeros((2*N,d)),    np.eye(2*N),                np.zeros((2*N,2*N)),       -maxAlloc*np.ones((2*N,1))]), # v_plus & v_minus < MaxAlloc*kappa
                   np.hstack([np.zeros((2*N,d)),    -np.eye(2*N),               np.zeros((2*N,2*N)),       np.zeros((2*N,1))]),          # v_plus, v_minus >= 0
                   np.hstack([np.zeros((2*N,d)),    np.zeros((2*N,2*N)),        -np.eye(2*N),              np.zeros((2*N,1))])])         # v_tou_plus, v_tou_minus >= 0
    h = np.hstack([np.zeros(2), -1e-12, np.zeros(6*N)])            # Bounds for inequalities

    #                         w_tou                 v_plus, v_minus             v_tou_plus, v_tou_minus    k
    A = np.vstack([np.hstack([V,                    -np.eye(N), np.eye(N),      np.zeros((N,2*N)),         np.zeros((N,1))]), # V w_tou = w'a_i = v_plus - v_minus
                   np.hstack([np.sum(V,axis=0),     np.zeros(2*N),              np.zeros(2*N),             -1]),              # sum w_a_i / kappa = 1, sum to 1 constraint
                   np.hstack([meanVec,              np.zeros(2*N),              np.zeros(2*N),             0]),               # w_tou = kappa w (if risk-free = 0)
                   np.hstack([-V,                   np.zeros((N,2*N)),          np.eye(N), -1*np.eye(N),   w_prev])           # V_tou = v_plus - v_minus = kappa (V w_f - w_prev)
                   ])
    b = np.hstack([np.zeros(N), 0, 1, np.zeros(N)])                # Equality constraint value

    # Expand sigma matrix, this is a square matrix
    target_dim = A.shape[1]
    sigMat_exp = np.vstack([
        np.hstack([sigMat, np.zeros((sigMat.shape[0],target_dim-sigMat.shape[1]))]),
        np.zeros((target_dim-sigMat.shape[0], target_dim))
    ])
    P = sigMat_exp

    q = np.hstack([np.zeros(d), lambda_l1 * np.ones(2 * N), lambda_l3*costVec, lambda_l3*costVec, 0])

    solvers.options['abstol'] = 1e-4 # Absolute tolerance
    solvers.options['reltol'] = 1e-4  # Relative tolerance
    solvers.options['feastol'] = 1e-4
    solvers.options['show_progress'] = False
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)
    sol = solvers.qp(P, q, G, h, A, b)
    # Solve problem
    res = np.array(sol['x']).flatten()

    # Process output to pd.Series where index names are asset id
    w_asset_opt = (res[d:d+N] - res[d+N:d+2*N]) / res[-1]
    
    return w_asset_opt

def portfolio_optimization_cvx_tc_v2(
        meanVec,sigMat,V,
        maxExposure,maxAlloc=1,
        lambda_l1=0,lambda_l2=0,lambda_l3=0,
        w_prev = None, costVec = None
):
    """
    Not sum to anything

    Parameters
    ----------
    sigMat: array-like
        Covariance matrix of assets
    LongShort : Float
        Takes value between 0 and 1, maximum total short position allowed
    maxAlloc : Float
        Takes value between 0 and 1. Specifies the maximum weight an asset can get
    lambda_l1 : Float
        Takes a value greater than 0. Specifies L1 penalty
    lambda_l2 : Float
        Takes a value greater than 0. Specifies L2 penalty
    lambda_l3 : Float
        Takes a value greater than 0. Specifies L3 penalty
    V: array-like
        Nxk matrix, corresponds to the V matrix in the paper
    meanVec: array-like, optional
        Mean return vector of assets, defaults to None
    w_prev: array-like, optional
        Previous weights of assets, default to 0

    Returns
    -------
    w_opt : Array
        Returns the weights of given to each asset in form of a numpy array
    var_opt : Float
        Returns the variance of the portfolio
    """
    # Process input
    meanVec = np.array(meanVec).flatten()
    sigMat = np.array(sigMat)
    V = np.array(V)
    w_prev = np.array(w_prev).flatten().reshape(-1, 1) # column matrix
    costVec = np.array(costVec)

    d = sigMat.shape[0]
    N = V.shape[0]

    sigMat = sigMatShrinkage(sigMat, lambda_l2, V)
    sigMat = nearestPD(sigMat)

    # G x <= h
    # A x = b
    # V here should be Nxk matrix, corresponds to V in the paper
    #                         w_tou                 v_plus, v_minus             v_tou_plus, v_tou_minus    k
    G = np.vstack([np.hstack([np.zeros(d),          np.ones(N), np.zeros(N),    np.zeros(2*N),             -(maxExposure/2)]),           # sum v_plus / kappa <= maxExposure / 2
                   np.hstack([np.zeros(d),          np.zeros(N), np.ones(N),    np.zeros(2*N),             -(maxExposure/2)]),           # sum v_minus / kappa <= maxExposure / 2
                   np.hstack([np.zeros(d),          np.zeros(2*N),              np.zeros(2*N),             -1]),                         # kappa > 0
                   np.hstack([np.zeros((2*N,d)),    np.eye(2*N),                np.zeros((2*N,2*N)),       -maxAlloc*np.ones((2*N,1))]), # v_plus & v_minus < MaxAlloc*kappa
                   np.hstack([np.zeros((2*N,d)),    -np.eye(2*N),               np.zeros((2*N,2*N)),       np.zeros((2*N,1))]),          # v_plus, v_minus >= 0
                   np.hstack([np.zeros((2*N,d)),    np.zeros((2*N,2*N)),        -np.eye(2*N),              np.zeros((2*N,1))])])         # v_tou_plus, v_tou_minus >= 0
    h = np.hstack([np.zeros(2), -1e-12, np.zeros(6*N)])            # Bounds for inequalities

    #                         w_tou                 v_plus, v_minus             v_tou_plus, v_tou_minus    k
    A = np.vstack([np.hstack([V,                    -np.eye(N), np.eye(N),      np.zeros((N,2*N)),         np.zeros((N,1))]), # V w_tou = w'a_i = v_plus - v_minus
                   np.hstack([meanVec,              np.zeros(2*N),              np.zeros(2*N),             0]),               # w_tou = kappa w (if risk-free = 0)
                   np.hstack([-V,                   np.zeros((N,2*N)),          np.eye(N), -1*np.eye(N),   w_prev])           # V_tou = v_plus - v_minus = kappa (V w_f - w_prev)
                   ])
    b = np.hstack([np.zeros(N), 1, np.zeros(N)])                # Equality constraint value

    # Expand sigma matrix, this is a square matrix
    target_dim = A.shape[1] # expand to number of variables to optimize for, can also be G.shape[1]
    sigMat_exp = np.vstack([
        np.hstack([sigMat, np.zeros((sigMat.shape[0],target_dim-sigMat.shape[1]))]),
        np.zeros((target_dim-sigMat.shape[0], target_dim))
    ])
    P = sigMat_exp

    q = np.hstack([np.zeros(d), lambda_l1 * np.ones(2 * N), lambda_l3*costVec, lambda_l3*costVec, 0])

    solvers.options['abstol'] = 1e-4 # Absolute tolerance
    solvers.options['reltol'] = 1e-4  # Relative tolerance
    solvers.options['feastol'] = 1e-4
    solvers.options['show_progress'] = False
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)
    sol = solvers.qp(P, q, G, h, A, b)
    # Solve problem
    res = np.array(sol['x']).flatten()

    # Process output to pd.Series where index names are asset id
    w_asset_opt = (res[d:d+N] - res[d+N:d+2*N]) / res[-1]
    
    return w_asset_opt

def portfolio_optimization_cvx_tc_v3(
        meanVec,sigMat,V,
        maxExposure,maxAlloc=1,
        lambda_l1=0,lambda_l2=0,lambda_l3=0,
        w_prev = None, costVec = None
):
    """
    No sum to anything constraint, targets return

    Parameters
    ----------
    sigMat: array-like
        Covariance matrix of assets
    LongShort : Float
        Takes value between 0 and 1, maximum total short position allowed
    maxAlloc : Float
        Takes value between 0 and 1. Specifies the maximum weight an asset can get
    lambda_l1 : Float
        Takes a value greater than 0. Specifies L1 penalty
    lambda_l2 : Float
        Takes a value greater than 0. Specifies L2 penalty
    lambda_l3 : Float
        Takes a value greater than 0. Specifies L3 penalty
    V: array-like
        Nxk matrix, corresponds to the V matrix in the paper
    meanVec: array-like, optional
        Mean return vector of assets, defaults to None
    w_prev: array-like, optional
        Previous weights of assets, default to 0

    Returns
    -------
    w_opt : Array
        Returns the weights of given to each asset in form of a numpy array
    var_opt : Float
        Returns the variance of the portfolio
    """
    # Process input
    meanVec = np.array(meanVec).flatten()
    sigMat = np.array(sigMat)
    V = np.array(V)
    w_prev = np.array(w_prev).flatten().reshape(-1, 1) # column matrix
    costVec = np.array(costVec)

    d = sigMat.shape[0]
    N = V.shape[0]

    sigMat = sigMatShrinkage(sigMat, lambda_l2, V)
    sigMat = nearestPD(sigMat)

    # G x <= h
    # A x = b
    # V here should be Nxk matrix, corresponds to V in the paper
    #                         w_tou                 v_plus, v_minus             v_tou_plus, v_tou_minus    k
    G = np.vstack([np.hstack([np.zeros(d),          np.ones(N), np.ones(N),     np.zeros(2*N),             -maxExposure]),               # sum (v_plus + v_minus) / kappa <= maxExposure
                   np.hstack([np.zeros(d),          np.zeros(2*N),              np.zeros(2*N),             -1]),                         # kappa > 0
                   np.hstack([np.zeros((2*N,d)),    np.eye(2*N),                np.zeros((2*N,2*N)),       -maxAlloc*np.ones((2*N,1))]), # v_plus & v_minus < MaxAlloc*kappa
                   np.hstack([np.zeros((2*N,d)),    -np.eye(2*N),               np.zeros((2*N,2*N)),       np.zeros((2*N,1))]),          # v_plus, v_minus >= 0
                   np.hstack([np.zeros((2*N,d)),    np.zeros((2*N,2*N)),        -np.eye(2*N),              np.zeros((2*N,1))])])         # v_tou_plus, v_tou_minus >= 0
    h = np.hstack([0, -1e-12, np.zeros(6*N)])            # Bounds for inequalities

    #                         w_tou                 v_plus, v_minus             v_tou_plus, v_tou_minus    k
    A = np.vstack([np.hstack([V,                    -np.eye(N), np.eye(N),      np.zeros((N,2*N)),         np.zeros((N,1))]), # V w_tou = w'a_i = v_plus - v_minus
                   np.hstack([meanVec,              np.zeros(2*N),              np.zeros(2*N),             0]),               # w_tou = kappa w (if risk-free = 0)
                   np.hstack([-V,                   np.zeros((N,2*N)),          np.eye(N), -1*np.eye(N),   w_prev]),          # V_tou = v_plus - v_minus = kappa (V w_f - w_prev)
                   np.hstack([meanVec,              np.zeros(2*N),              np.zeros(2*N),             -0.015])           # Target return = 1.5%
                   ])
    b = np.hstack([np.zeros(N), 1, np.zeros(N), 0])                # Equality constraint value

    # Expand sigma matrix, this is a square matrix
    target_dim = A.shape[1] # expand to number of variables to optimize for, can also be G.shape[1]
    sigMat_exp = np.vstack([
        np.hstack([sigMat, np.zeros((sigMat.shape[0],target_dim-sigMat.shape[1]))]),
        np.zeros((target_dim-sigMat.shape[0], target_dim))
    ])
    P = sigMat_exp

    q = np.hstack([np.zeros(d), lambda_l1 * np.ones(2 * N), lambda_l3*costVec, lambda_l3*costVec, 0])

    solvers.options['abstol'] = 1e-4 # Absolute tolerance
    solvers.options['reltol'] = 1e-4  # Relative tolerance
    solvers.options['feastol'] = 1e-4
    solvers.options['show_progress'] = False
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)
    sol = solvers.qp(P, q, G, h, A, b)
    # Solve problem
    res = np.array(sol['x']).flatten()

    # Process output to pd.Series where index names are asset id
    w_asset_opt = (res[d:d+N] - res[d+N:d+2*N]) / res[-1]
    
    return w_asset_opt


def portfolio_optimization_cvx_tc_online(
        meanVec,sigMat,V,
        longShort,maxAlloc=1,
        lambda_l1=0,lambda_l2=0,lambda_l3=0,
        w_prev = None, costVec = None
):
    """
    function do the portfolio optimization

    Parameters
    ----------
    sigMat: array-like
        Covariance matrix of assets
    LongShort : Float
        Takes value between 0 and 1, maximum total short position allowed
    maxAlloc : Float
        Takes value between 0 and 1. Specifies the maximum weight an asset can get
    lambda_l1 : Float
        Takes a value greater than 0. Specifies L1 penalty
    lambda_l2 : Float
        Takes a value greater than 0. Specifies L2 penalty
    lambda_l3 : Float
        Takes a value greater than 0. Specifies L3 penalty
    V: array-like
        Nxk matrix, corresponds to the V matrix in the paper
    meanVec: array-like, optional
        Mean return vector of assets, defaults to None
    w_prev: array-like, optional
        Previous weights of assets, default to 0

    Returns
    -------
    w_opt : Array
        Returns the weights of given to each asset in form of a numpy array
    var_opt : Float
        Returns the variance of the portfolio
    """
    # Process input
    meanVec = np.array(meanVec).flatten()
    sigMat = np.array(sigMat)
    V = np.array(V)
    w_prev = np.array(w_prev).flatten().reshape(-1, 1) # column matrix
    costVec = np.array(costVec)

    d = sigMat.shape[0]
    N = V.shape[0]

    sigMat = sigMatShrinkage(sigMat, lambda_l2, V)
    sigMat = nearestPD(sigMat)

    # G x <= h
    # A x = b
    # V here should be Nxk matrix, corresponds to V in the paper
    #                         w_tou                 v_plus, v_minus             v_tou_plus, v_tou_minus    k
    G = np.vstack([np.hstack([np.zeros(d),          np.ones(N), np.zeros(N),    np.zeros(2*N),             -(1+longShort)]),             # sum v_plus / kappa <= (1+LongShort)
                   np.hstack([np.zeros(d),          np.zeros(N), np.ones(N),    np.zeros(2*N),             -longShort]),                 # sum v_minus / kappa <= LongShort
                   np.hstack([np.zeros(d),          np.zeros(2*N),              np.zeros(2*N),             -1]),                         # kappa > 0
                   np.hstack([np.zeros((2*N,d)),    np.eye(2*N),                np.zeros((2*N,2*N)),       -maxAlloc*np.ones((2*N,1))]), # v_plus & v_minus < MaxAlloc*kappa
                   np.hstack([np.zeros((2*N,d)),    -np.eye(2*N),               np.zeros((2*N,2*N)),       np.zeros((2*N,1))]),          # v_plus, v_minus >= 0
                   np.hstack([np.zeros((2*N,d)),    np.zeros((2*N,2*N)),        -np.eye(2*N),              np.zeros((2*N,1))])])         # v_tou_plus, v_tou_minus >= 0
    h = np.hstack([np.zeros(2), -1e-12, np.zeros(6*N)])            # Bounds for inequalities

    #                         w_tou                 v_plus, v_minus             v_tou_plus, v_tou_minus    k
    A = np.vstack([np.hstack([V,                    -np.eye(N), np.eye(N),      np.zeros((N,2*N)),         np.zeros((N,1))]), # V w_tou = w'a_i = v_plus - v_minus
                   np.hstack([np.sum(V,axis=0),     np.zeros(2*N),              np.zeros(2*N),             -1]),              # sum w_a_i / kappa = 1, sum to 1 constraint
                   np.hstack([meanVec,              np.zeros(2*N),              np.zeros(2*N),             0]),               # w_tou = kappa w (if risk-free = 0)
                   np.hstack([-V,                   np.zeros((N,2*N)),          np.eye(N), -1*np.eye(N),   w_prev])           # V_tou = v_plus - v_minus = kappa (V w_f - w_prev)
                   ])
    b = np.hstack([np.zeros(N), 0, 1, np.zeros(N)])                # Equality constraint value

    # Expand sigma matrix, this is a square matrix
    target_dim = A.shape[1]
    sigMat_exp = np.vstack([
        np.hstack([sigMat, np.zeros((sigMat.shape[0],target_dim-sigMat.shape[1]))]),
        np.zeros((target_dim-sigMat.shape[0], target_dim))
    ])
    P = sigMat_exp

    q = np.hstack([np.zeros(d), lambda_l1 * np.ones(2 * N), lambda_l3*costVec, lambda_l3*costVec, 0])

    solvers.options['abstol'] = 1e-4 # Absolute tolerance
    solvers.options['reltol'] = 1e-4  # Relative tolerance
    solvers.options['feastol'] = 1e-4
    solvers.options['show_progress'] = False
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)
    sol = solvers.qp(P, q, G, h, A, b)
    # Solve problem
    res = np.array(sol['x']).flatten()

    # Process output to pd.Series where index names are asset id
    w_asset_opt = (res[d:d+N] - res[d+N:d+2*N]) / res[-1]

    w_factor_opt = res[0:d]
    
    return w_asset_opt, w_factor_opt


def portfolio_optimization_cvx_tc_beta(
        meanVec,sigMat,V,
        maxExposure,maxAlloc=1,betaTarget=1,
        lambda_l1=0,lambda_l2=0,lambda_l3=0,
        w_prev = None, costVec = None, betaVec = None
):
    """
    Not sum to anything, beta target

    Parameters
    ----------
    sigMat: array-like
        Covariance matrix of assets
    LongShort : Float
        Takes value between 0 and 1, maximum total short position allowed
    maxAlloc : Float
        Takes value between 0 and 1. Specifies the maximum weight an asset can get
    lambda_l1 : Float
        Takes a value greater than 0. Specifies L1 penalty
    lambda_l2 : Float
        Takes a value greater than 0. Specifies L2 penalty
    lambda_l3 : Float
        Takes a value greater than 0. Specifies L3 penalty
    V: array-like
        Nxk matrix, corresponds to the V matrix in the paper
    meanVec: array-like, optional
        Mean return vector of assets, defaults to None
    w_prev: array-like, optional
        Previous weights of assets, default to 0

    Returns
    -------
    w_opt : Array
        Returns the weights of given to each asset in form of a numpy array
    var_opt : Float
        Returns the variance of the portfolio
    """
    # Process input
    meanVec = np.array(meanVec).flatten()
    sigMat = np.array(sigMat)
    V = np.array(V)
    w_prev = np.array(w_prev).flatten().reshape(-1, 1) # column matrix
    costVec = np.array(costVec)

    d = sigMat.shape[0]
    N = V.shape[0]

    sigMat = sigMatShrinkage(sigMat, lambda_l2, V)
    sigMat = nearestPD(sigMat)

    # G x <= h
    # A x = b
    # V here should be Nxk matrix, corresponds to V in the paper
    #                         w_tou                 v_plus, v_minus             v_tou_plus, v_tou_minus    k
    G = np.vstack([np.hstack([np.zeros(d),          np.ones(N), np.zeros(N),    np.zeros(2*N),             -(maxExposure/2)]),           # sum v_plus / kappa <= maxExposure / 2
                   np.hstack([np.zeros(d),          np.zeros(N), np.ones(N),    np.zeros(2*N),             -(maxExposure/2)]),           # sum v_minus / kappa <= maxExposure / 2
                   np.hstack([np.zeros(d),          np.zeros(2*N),              np.zeros(2*N),             -1]),                         # kappa > 0
                   np.hstack([np.zeros((2*N,d)),    np.eye(2*N),                np.zeros((2*N,2*N)),       -maxAlloc*np.ones((2*N,1))]), # v_plus & v_minus < MaxAlloc*kappa
                   np.hstack([np.zeros((2*N,d)),    -np.eye(2*N),               np.zeros((2*N,2*N)),       np.zeros((2*N,1))]),          # v_plus, v_minus >= 0
                   np.hstack([np.zeros((2*N,d)),    np.zeros((2*N,2*N)),        -np.eye(2*N),              np.zeros((2*N,1))])])         # v_tou_plus, v_tou_minus >= 0
    h = np.hstack([np.zeros(2), -1e-12, np.zeros(6*N)])            # Bounds for inequalities

    #                         w_tou                 v_plus, v_minus             v_tou_plus, v_tou_minus    k
    A = np.vstack([np.hstack([V,                    -np.eye(N), np.eye(N),      np.zeros((N,2*N)),         np.zeros((N,1))]), # V w_tou = w'a_i = v_plus - v_minus
                   np.hstack([meanVec,              np.zeros(2*N),              np.zeros(2*N),             0]),               # w_tou = kappa w (if risk-free = 0)
                   np.hstack([-V,                   np.zeros((N,2*N)),          np.eye(N), -1*np.eye(N),   w_prev]),          # V_tou = v_plus - v_minus = kappa (V w_f - w_prev)
                   np.hstack([np.zeros(d),          1*betaVec, -1*betaVec,      np.zeros(2*N),             -betaTarget])      # beta = betaTarget
                   ])
    b = np.hstack([np.zeros(N), 1, np.zeros(N), 0])                # Equality constraint value

    # Expand sigma matrix, this is a square matrix
    target_dim = A.shape[1] # expand to number of variables to optimize for, can also be G.shape[1]
    sigMat_exp = np.vstack([
        np.hstack([sigMat, np.zeros((sigMat.shape[0],target_dim-sigMat.shape[1]))]),
        np.zeros((target_dim-sigMat.shape[0], target_dim))
    ])
    P = sigMat_exp

    q = np.hstack([np.zeros(d), lambda_l1 * np.ones(2 * N), lambda_l3*costVec, lambda_l3*costVec, 0])

    solvers.options['abstol'] = 1e-4 # Absolute tolerance
    solvers.options['reltol'] = 1e-4  # Relative tolerance
    solvers.options['feastol'] = 1e-4
    solvers.options['show_progress'] = False
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)
    sol = solvers.qp(P, q, G, h, A, b)
    # Solve problem
    res = np.array(sol['x']).flatten()

    # Process output to pd.Series where index names are asset id
    w_asset_opt = (res[d:d+N] - res[d+N:d+2*N]) / res[-1]
    
    return w_asset_opt


def portfolio_optimization_cvx_tc_free(
        meanVec,sigMat,V,
        targetExposure=2,
        lambda_l1=0,lambda_l2=0,lambda_l3=0,
        w_prev = None, costVec = None
):
    """
    No constraint

    Parameters
    ----------
    sigMat: array-like
        Covariance matrix of assets
    LongShort : Float
        Takes value between 0 and 1, maximum total short position allowed
    maxAlloc : Float
        Takes value between 0 and 1. Specifies the maximum weight an asset can get
    lambda_l1 : Float
        Takes a value greater than 0. Specifies L1 penalty
    lambda_l2 : Float
        Takes a value greater than 0. Specifies L2 penalty
    lambda_l3 : Float
        Takes a value greater than 0. Specifies L3 penalty
    V: array-like
        Nxk matrix, corresponds to the V matrix in the paper
    meanVec: array-like, optional
        Mean return vector of assets, defaults to None
    w_prev: array-like, optional
        Previous weights of assets, default to 0

    Returns
    -------
    w_opt : Array
        Returns the weights of given to each asset in form of a numpy array
    var_opt : Float
        Returns the variance of the portfolio
    """
    # Process input
    meanVec = np.array(meanVec).flatten()
    sigMat = np.array(sigMat)
    V = np.array(V)
    w_prev = np.array(w_prev).flatten().reshape(-1, 1) # column matrix
    costVec = np.array(costVec)

    d = sigMat.shape[0]
    N = V.shape[0]

    sigMat = sigMatShrinkage(sigMat, lambda_l2, V)
    sigMat = nearestPD(sigMat)

    # G x <= h
    # A x = b
    # V here should be Nxk matrix, corresponds to V in the paper
    #                         w_tou                 v_plus, v_minus             v_tou_plus, v_tou_minus    k
    G = np.vstack([np.hstack([np.zeros(d),          np.zeros(2*N),              np.zeros(2*N),             -1]),                         # kappa > 0
                   np.hstack([np.zeros((2*N,d)),    -np.eye(2*N),               np.zeros((2*N,2*N)),       np.zeros((2*N,1))]),          # v_plus, v_minus >= 0
                   np.hstack([np.zeros((2*N,d)),    np.zeros((2*N,2*N)),        -np.eye(2*N),              np.zeros((2*N,1))])])         # v_tou_plus, v_tou_minus >= 0
    h = np.hstack([-1e-12, np.zeros(4*N)])            # Bounds for inequalities

    #                         w_tou                 v_plus, v_minus             v_tou_plus, v_tou_minus    k
    A = np.vstack([np.hstack([V,                    -np.eye(N), np.eye(N),      np.zeros((N,2*N)),         np.zeros((N,1))]), # V w_tou = w'a_i = v_plus - v_minus
                   np.hstack([meanVec,              np.zeros(2*N),              np.zeros(2*N),             0]),               # w_tou = kappa w (if risk-free = 0)
                   np.hstack([-V,                   np.zeros((N,2*N)),          np.eye(N), -1*np.eye(N),   w_prev]),          # V_tou = v_plus - v_minus = kappa (V w_f - w_prev)
                   np.hstack([np.zeros(d),          np.ones(N), np.ones(N),     np.zeros(2*N),             -targetExposure]), # sum (v_plus + v_minus) / kappa = targetExposure
                   ])
    b = np.hstack([np.zeros(N), 1, np.zeros(N), 0])                # Equality constraint value

    # Expand sigma matrix, this is a square matrix
    target_dim = A.shape[1] # expand to number of variables to optimize for, can also be G.shape[1]
    sigMat_exp = np.vstack([
        np.hstack([sigMat, np.zeros((sigMat.shape[0],target_dim-sigMat.shape[1]))]),
        np.zeros((target_dim-sigMat.shape[0], target_dim))
    ])
    P = sigMat_exp

    q = np.hstack([np.zeros(d), lambda_l1 * np.ones(2 * N), lambda_l3*costVec, lambda_l3*costVec, 0])

    solvers.options['abstol'] = 1e-4 # Absolute tolerance
    solvers.options['reltol'] = 1e-4  # Relative tolerance
    solvers.options['feastol'] = 1e-4
    solvers.options['show_progress'] = False
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)
    sol = solvers.qp(P, q, G, h, A, b)
    # Solve problem
    res = np.array(sol['x']).flatten()

    # Process output to pd.Series where index names are asset id
    w_asset_opt = (res[d:d+N] - res[d+N:d+2*N]) / res[-1]
    
    return w_asset_opt


def portfolio_optimization_cvx_tc_tangency(
        meanVec,sigMat,V,
        targetExposure=2,
        lambda_l1=0,lambda_l2=0,lambda_l3=0,
        w_prev = None, costVec = None
):
    """
    No constraint

    Parameters
    ----------
    sigMat: array-like
        Covariance matrix of assets
    LongShort : Float
        Takes value between 0 and 1, maximum total short position allowed
    maxAlloc : Float
        Takes value between 0 and 1. Specifies the maximum weight an asset can get
    lambda_l1 : Float
        Takes a value greater than 0. Specifies L1 penalty
    lambda_l2 : Float
        Takes a value greater than 0. Specifies L2 penalty
    lambda_l3 : Float
        Takes a value greater than 0. Specifies L3 penalty
    V: array-like
        Nxk matrix, corresponds to the V matrix in the paper
    meanVec: array-like, optional
        Mean return vector of assets, defaults to None
    w_prev: array-like, optional
        Previous weights of assets, default to 0

    Returns
    -------
    w_opt : Array
        Returns the weights of given to each asset in form of a numpy array
    var_opt : Float
        Returns the variance of the portfolio
    """
    # Process input
    meanVec = np.array(meanVec).flatten()
    sigMat = np.array(sigMat)
    V = np.array(V)
    w_prev = np.array(w_prev).flatten().reshape(-1, 1) # column matrix
    costVec = np.array(costVec)

    d = sigMat.shape[0]
    N = V.shape[0]

    sigMat = sigMatShrinkage(sigMat, lambda_l2, V)
    sigMat = nearestPD(sigMat)

    # G x <= h
    # A x = b
    # V here should be Nxk matrix, corresponds to V in the paper
    #                         w_tou                 v_plus, v_minus             v_tou_plus, v_tou_minus    k
    G = np.vstack([np.hstack([np.zeros(d),          np.zeros(2*N),              np.zeros(2*N),             -1]),                         # kappa > 0
                   np.hstack([np.zeros((2*N,d)),    -np.eye(2*N),               np.zeros((2*N,2*N)),       np.zeros((2*N,1))]),          # v_plus, v_minus >= 0
                   np.hstack([np.zeros((2*N,d)),    np.zeros((2*N,2*N)),        -np.eye(2*N),              np.zeros((2*N,1))])])         # v_tou_plus, v_tou_minus >= 0
    h = np.hstack([-1e-12, np.zeros(4*N)])            # Bounds for inequalities

    #                         w_tou                 v_plus, v_minus             v_tou_plus, v_tou_minus    k
    A = np.vstack([np.hstack([V,                    -np.eye(N), np.eye(N),      np.zeros((N,2*N)),         np.zeros((N,1))]), # V w_tou = w'a_i = v_plus - v_minus
                   np.hstack([meanVec,              np.zeros(2*N),              np.zeros(2*N),             0]),               # w_tou = kappa w (if risk-free = 0)
                   np.hstack([-V,                   np.zeros((N,2*N)),          np.eye(N), -1*np.eye(N),   w_prev]),          # V_tou = v_plus - v_minus = kappa (V w_f - w_prev)
                   #np.hstack([np.zeros(d),          np.ones(N), np.ones(N),     np.zeros(2*N),             -targetExposure]), # sum (v_plus + v_minus) / kappa = targetExposure
                   np.hstack([np.ones(d),           np.zeros(N), np.zeros(N),   np.zeros(2*N),             -1]),              # sum of factor weights = 1
                   ])
    b = np.hstack([np.zeros(N), 1, np.zeros(N), 0])                # Equality constraint value

    # Expand sigma matrix, this is a square matrix
    target_dim = A.shape[1] # expand to number of variables to optimize for, can also be G.shape[1]
    sigMat_exp = np.vstack([
        np.hstack([sigMat, np.zeros((sigMat.shape[0],target_dim-sigMat.shape[1]))]),
        np.zeros((target_dim-sigMat.shape[0], target_dim))
    ])
    P = sigMat_exp

    q = np.hstack([np.zeros(d), lambda_l1 * np.ones(2 * N), lambda_l3*costVec, lambda_l3*costVec, 0])

    solvers.options['abstol'] = 1e-4 # Absolute tolerance
    solvers.options['reltol'] = 1e-4  # Relative tolerance
    solvers.options['feastol'] = 1e-4
    solvers.options['show_progress'] = False
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)
    sol = solvers.qp(P, q, G, h, A, b)
    # Solve problem
    res = np.array(sol['x']).flatten()

    # Process output to pd.Series where index names are asset id
    w_asset_opt = (res[d:d+N] - res[d+N:d+2*N]) / res[-1]
    
    return w_asset_opt