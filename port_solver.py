import numpy as np
from cvxopt import solvers
from cvxopt import matrix
from numpy import linalg as la

def svd_modify_diagonal(matrix):
    """
    Perform SVD on a matrix, add a small value to the diagonal terms of Sigma,
    and optionally reconstruct the matrix using the modified Sigma.

    Parameters:
    matrix (np.ndarray): The input matrix to perform SVD on.
    small_value (float): The small value to add to the diagonal terms of Sigma.
    reconstruct (bool): Whether to reconstruct and return the modified matrix.

    Returns:
    U (np.ndarray): Left singular vectors.
    Sigma_modified (np.ndarray): Modified singular values with the small value added.
    VT (np.ndarray): Right singular vectors (transposed).
    A_modified (np.ndarray, optional): Reconstructed matrix using modified Sigma.
    """
 
    # Perform SVD
    U, Sigma, VT = np.linalg.svd(matrix)
    
    # Add small value to the singular values
    Sigma_modified = np.where(Sigma < 1e-4, Sigma + 1e-4, Sigma)
    
    # Optional: Reconstruct the matrix with modified Sigma
    A_modified = None

    # Create a diagonal matrix of zeros with dimensions matching the original matrix
    m, n = matrix.shape
    Sigma_matrix = np.zeros((m, n))
    np.fill_diagonal(Sigma_matrix, Sigma_modified)
    
    # Reconstruct the matrix with the modified Sigma
    A_modified = np.dot(U, np.dot(Sigma_matrix, VT))

    return A_modified


def sigMatShrinkage(sigMat,lambda_l2, factor = None):
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

        
def penalty_vector(d, N, sigMat, maxShar = 0, factor = None, longShort=0, lambda_l1=0, turnover = None, exposure_constrain = 0, TE_constrain = None, Q_b = None, Q_bench = None):
    """
    This function calculates the penalty vector for an optimization problem, given a set of parameters. 
    
    Parameters:
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
    
    Returns:
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
    
    Parameters:
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
    
    Returns:
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
    
    h = np.hstack([np.zeros(2*N+2), -1e-12, np.zeros(4*N)])            # Example bounds for inequalities
    G = np.vstack([np.hstack([factor, np.zeros((N,2*N)), -0.08*np.ones((N,1))]), # -inf < w'f_i <U
                               np.hstack([np.zeros(d), np.ones(N), np.zeros(N), -1.2]), # sum u_plus <= kappa(1+longshort)
                               np.hstack([np.zeros(d), np.zeros(N), np.ones(N), -0.2]),
                               np.hstack([-factor, np.zeros((N,2*N)), -0.08*np.ones((N,1))]), # L < w'f_i < inf
                               np.hstack([np.zeros(d+2*N), -1]), # kappa > 0
                               np.hstack([np.zeros((2*N,d)), np.eye(2*N), -0.08*np.ones((2*N,1))]), # -inf < u_plus&u_minus < 0.08*kappa
                               np.hstack([np.zeros((2*N,d)), -np.eye(2*N), np.zeros((2*N,1))])])
    b = np.hstack([np.zeros(N+1), 1])                    # Example equality constraint value
    A = np.vstack([np.hstack([factor, -np.eye(N), np.eye(N), np.zeros((N,1))]), #w'f_i = u_plus - u_minus
                               np.hstack([np.sum(factor,axis=0), np.zeros(2*N), -1]),# sum w'f = kapp
                               np.hstack([meanVec, np.zeros(2*N), 0])
                               ]) 

    P = sigMat_expend(d, N, sigMat_shrik, maxShar, factor, longShort, turnover, exposure_constrain, TE_constrain, general_quad, Q_w, Q_b)
    q = penalty_vector(d, N, sigMat_shrik, maxShar, factor, longShort, lambda_l1, turnover, exposure_constrain, TE_constrain, Q_b, Q_bench)
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)
    solvers.options['abstol'] = 1e-4 # Absolute tolerance
    solvers.options['reltol'] = 1e-4  # Relative tolerance
    solvers.options['feastol'] = 1e-4
    sol = solvers.qp(P, q, G, h, A, b)
    # Solve problem
    w_opt = np.array(sol['x'])
    # test_individual = np.dot(factor, w_opt[:d].reshape(-1,1))/w_opt[-1]
    # print('maximum')
    # print(max(test_individual))
    # print('long')
    # print(np.sum(test_individual[test_individual > 0]))
    # print('positive')
    # print(np.sum(w_opt[d:d+N]/w_opt[-1]))
    # print('negative')
    # print(np.sum(w_opt[d+N:d+2*N]/w_opt[-1]))
    # print('kappa')
    # print(w_opt[-1])
    
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
    # Var_opt = np.dot(np.dot(w_opt, sigMat), w_opt.transpose())
    if assetsOrder:
        w_opt = w_opt[assetsOrder]
    
    return w_opt