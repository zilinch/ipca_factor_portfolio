import numpy as np
import pandas as pd
from scipy.linalg import qr, inv, eig
from scipy.linalg import sqrtm
from . import pyport
from ..data_processing import data_helper



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


def RPPCAOOS(Xtotal, stdnorm, gamma, K, window):
    """
    Conduct Out-of-sample evaluation of Regularized Principal Component Analysis (RP-PCA).

    Parameters
    ----------
    Xtotal : 2D numpy array
        The input data where each row is a data point, each column is a feature.

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
    SR_grid : list
        Sharpe ratios for different combinations of the tuning parameters for RP-PCA.

    maxreturntime : list
        Returns for different combinations of the tuning parameters for RP-PCA.

    SR_grid_pca : list
        Sharpe ratios for different combinations of the tuning parameters for standard PCA.

    maxreturntime_pca : list
        Returns for different combinations of the tuning parameters for standard PCA.

    """

    N = Xtotal.shape[1]
    Ttotal = Xtotal.shape[0]

    X = Xtotal
    T = Ttotal

    # Initialize lists to store intermediate results
    Sigmatime, mutime, Sigmatime_pca, mutime_pca = [], [], [], []
    Lambdahatprevious_PCA = None
    Lambdahatprevious_RPPCA = None

    maxreturntime = [[None for _ in range(K)] for _ in range(100)] 
    maxreturntime_pca = [[None for _ in range(K)] for _ in range(100)] 

    for t in range(window, N):
        print(t)
        # Define the current window of data
        X = Xtotal[t-window:t, :]
        T = len(X[:, 0])
        Xnext = Xtotal[t, :]

        # Compute the RP-PCA factors for the current window
        Fhat_RPPCA, lambdahat_RPPCA, Lnext_RPPCA = PRPCA_factor(X, T, N, K, gamma, stdnorm, t-window, Lambdahatprevious_RPPCA)
        # Compute the standard PCA factors for the current window
        Fhat, lambdahat, Lnext_PCA = PCA_factor(X, K, stdnorm, t-window, Lambdahatprevious_PCA)

        Lambdahatprevious_PCA = Lnext_PCA
        Lambdahatprevious_RPPCA = Lnext_RPPCA

        maxreturntime_temp = [None] * (K-1)
        maxreturntime_temp_pca = [None] * (K-1) 

        # Compute the covariance and mean of the factors
        Sigmatime.append(np.cov(Fhat_RPPCA.T))
        mutime.append(np.mean(Fhat_RPPCA, axis=0))
        Sigmatime_pca.append(np.cov(Fhat.T))
        mutime_pca.append(np.mean(Fhat, axis=0))
        k = t-window
        g1 = np.exp(np.linspace( np.log(1e-6),np.log(5),10))
        g2 = np.exp(np.linspace( np.log(1e-6),np.log(5),10))
        # Loop through different combinations of tuning parameters
        for a in range(10):
                for b in range(10):
                    for i in range(1,K):
                        # Compute the optimal portfolio weights using RP-PCA and standard PCA
                        OptimalPortfolioWeightstime_temp = (np.hstack([pyport.portfolio_optimization(meanVec = np.array(mutime[k][0:i+1]),sigMat=np.array(Sigmatime[k][0:i+1, 0:i+1]),retTarget = 0,longShort = 0.2,maxAlloc=0.08,
                                                                        lambda_l1=g1[a],lambda_l2=g2[b],riskfree = 0,assetsOrder=None,maxShar = 1,factor=np.array(lambdahat_RPPCA)[:,0:i+1],turnover = None, 
                                                                        w_pre = None, individual = False, exposure_constrain = 0, w_bench = None, 
                                                                        factor_exposure_constrain = None, U_factor = None, general_linear_constrain = None, 
                                                                        U_genlinear = 0, w_general = None, TE_constrain = 0, general_quad = 0, Q_w = None, 
                                                                        Q_b = None, Q_bench = None)[0], np.zeros(K-i-1)])).reshape(-1,1)
                        
                        OptimalPortfolioWeightstime_temp_pca = (np.hstack([pyport.portfolio_optimization(meanVec = np.array(mutime_pca[k][0:i+1]),sigMat=np.array(Sigmatime_pca[k][0:i+1, 0:i+1]),retTarget = 0,longShort = 0.2,maxAlloc=0.08,
                                                                        lambda_l1=g1[a],lambda_l2=g2[b],riskfree = 0,assetsOrder=None,maxShar = 1,factor=np.array(lambdahat)[:,0:i+1],turnover = None, 
                                                                        w_pre = None, individual = False, exposure_constrain = 0, w_bench = None, 
                                                                        factor_exposure_constrain = None, U_factor = None, general_linear_constrain = None, 
                                                                        U_genlinear = 0, w_general = None, TE_constrain = 0, general_quad = 0, Q_w = None, 
                                                                        Q_b = None, Q_bench = None)[0], np.zeros(K-i-1)])).reshape(-1,1)
                        
                
                        ind = i-1
                        # Store the returns for different combinations of tuning parameters
                        if maxreturntime_temp[ind] is None:
                            maxreturntime_temp[ind] = Xnext @ lambdahat_RPPCA @ inv(lambdahat_RPPCA.T @ lambdahat_RPPCA) @ OptimalPortfolioWeightstime_temp
                            maxreturntime_temp_pca[ind] = Xnext @ lambdahat @ inv(lambdahat.T @ lambdahat) @ OptimalPortfolioWeightstime_temp_pca
                            
                        else:
                            maxreturntime_temp[ind] = np.vstack([maxreturntime_temp[ind], Xnext @ lambdahat_RPPCA @ inv(lambdahat_RPPCA.T @ lambdahat_RPPCA) @ OptimalPortfolioWeightstime_temp])
                            maxreturntime_temp_pca[ind] = np.vstack([maxreturntime_temp_pca[ind], Xnext @ lambdahat @ inv(lambdahat.T @ lambdahat) @ OptimalPortfolioWeightstime_temp_pca])
                            

                        if maxreturntime[a * 10 + b][ind] is None:
                            maxreturntime[a * 10 + b][ind] = maxreturntime_temp[ind][a * 10 + b]
                            maxreturntime_pca[a * 10 + b][ind] = maxreturntime_temp_pca[ind][a * 10 + b]
                            
                        else:
                            maxreturntime[a * 10 + b][ind] = np.vstack([maxreturntime[a * 10 + b][ind], maxreturntime_temp[ind][a * 10 + b]])
                            maxreturntime_pca[a * 10 + b][ind] = np.vstack([maxreturntime_pca[a * 10 + b][ind], maxreturntime_temp_pca[ind][a * 10 + b]])
                            
    # Compute the Sharpe ratios for different combinations of tuning parameters
    SR_grid = [[None for _ in range(K-1)] for _ in range(100)]
    SR_grid_pca = [[None for _ in range(K-1)] for _ in range(100)]
    
    for i in range(len(g1)*len(g2)):
        for j in range(K-1):
            maxreturnnormalizedtime = maxreturntime[i][j] / np.std(maxreturntime[i][j], axis=0)
            SR_grid[i][j]= np.mean(maxreturnnormalizedtime, axis=0)

            maxreturnnormalizedtime_pca = maxreturntime_pca[i][j] / np.std(maxreturntime_pca[i][j], axis=0)
            SR_grid_pca[i][j]= np.mean(maxreturnnormalizedtime_pca, axis=0)
    return SR_grid, maxreturntime, SR_grid_pca, maxreturntime_pca


if __name__ == "__main__":
    data = pd.read_csv("asness_final_revisedmkt.csv")
    data = data.drop(['Unnamed: 0','spread'], axis = 1)
    df = data.copy(deep = True)
    df_pre = data.loc[:, (data != 0).any(axis=0)]
    characteristics = df_pre.drop(['date','ret','prc','cusip','permco'], axis = 1).columns
    Xtotal = data_helper.single_sort(data, characteristics)


    shar_rppca, ret_rppca, shar_pca, ret_pca, = RPPCAOOS(np.array(Xtotal), stdnorm= 0, gamma = 15, K=6, window=240)


