import pandas as pd
import numpy as np
from numpy import linalg as LA
from datetime import datetime

from PortOpt_factor.data_processing.imputation import rolling_sum_list, rolling_sum, my_inverse



def backward_cross_section_imputation(C):
    
    dates = sorted(C['eom'].unique())
    K = 5

    # Lists to store intermediate results
    list_first_part = []
    list_second_part = []

    # Loop through each date
    for t in range(1, len(dates)):
        print(dates[t])
        # Subset the data for the current and previous date
        C_t = C[C['eom'] == dates[t]]
        C_t_pre = C[C['eom'] == dates[t-1]]
        # Get the unique stocks for the current date
        stocks = C_t['id'].unique()
        N_t = len(stocks)
        L = len(C_t.columns) - 6

        result_t = np.zeros((N_t, N_t))
        W_t_i = np.zeros((N_t, L))

        # Compute pairwise similarities between stocks based on their characteristics
        for i in range(N_t):
            for j in range(i, N_t):
                characteristics_i = C_t.loc[C_t['id'] == stocks[i]].drop(["id", "eom", "ret_exc_lead1m", "ret_local_lead1m", "ret_local", "ret_exc"], axis=1)
                characteristics_j = C_t.loc[C_t['id'] == stocks[j]].drop(["id", "eom", "ret_exc_lead1m", "ret_local_lead1m", "ret_local", "ret_exc"], axis=1)
                Q_t_ij = np.logical_and(characteristics_i.notna().values, characteristics_j.notna().values)
                W_t_i[i, :] = characteristics_i.notna().values

                if Q_t_ij.sum() == 0:
                    result_t[i,j] = 0
                    result_t[j,i] = 0
                else:
                    product_ij = characteristics_i.values * characteristics_j.values
                    product_ij[np.isnan(product_ij)] = 0
                    sum_product = product_ij.sum()
                    result_t[i, j] = sum_product / Q_t_ij.sum()
                    result_t[j, i] = result_t[i, j]

        # Compute eigenvalues and eigenvectors of the similarity matrix
        eig_t = LA.eig(result_t)
        index = np.argsort(eig_t[0])[::-1]
        index = index[:K]
        F_hat_t = np.real(eig_t[1][:, index])

        first_part_t = [np.zeros((K+1, K+1)) for _ in range(L)]
        second_part_t = np.zeros((K+1, L))

        # Compute the first and second parts of the imputation formula
        for j in range(L):
            for i in range(N_t):
                C_t_pre_i = C_t_pre.loc[C_t_pre['id'] == stocks[i], C_t.columns[j+3]].values if stocks[i] in C_t_pre['id'].values and pd.notna(C_t_pre.loc[C_t_pre['id'] == stocks[i], C_t.columns[j+3]]).values else 0
                X_il_t = np.insert(F_hat_t[i, :], 0, C_t_pre_i).reshape(-1, 1)
                first_part_t[j] = first_part_t[j] + W_t_i[i,j] * X_il_t @ X_il_t.T
                C_t_i = C_t.loc[C_t['id'] == stocks[i], C_t.columns[j+3]].values if pd.notna(C_t.loc[C_t['id'] == stocks[i], C_t.columns[j+3]]).values else 0
                second_part_t[:, j] = second_part_t[:, j] + W_t_i[i,j] * X_il_t.flatten() * C_t_i

        list_first_part.append(first_part_t)
        list_second_part.append(second_part_t)
        
        # Compute rolling sums for the first and second parts
        first_part = rolling_sum_list(list_first_part, t-1, 12)[0]
        second_part = rolling_sum(list_second_part, t-1, 12)
        beta = np.zeros((K+1, L))
        
        # Compute beta coefficients for imputation
        for j in range(L):                    
            if not np.any(first_part[j]):
                beta[:,j] = np.zeros(K+1)
            else:
                beta[:,j] = my_inverse(first_part[j]) @ second_part[:,j]
            
            # Identify missing values and impute them
            na_index = np.where(pd.isna(C_t.iloc[:,j+3]))[0]
            for i in na_index:
                C_t_pre_i = C_t_pre.loc[C_t_pre['id'] == stocks[i], C_t.columns[j+3]].values if stocks[i] in C_t_pre['id'].values and pd.notna(C_t_pre.loc[C_t_pre['id'] == stocks[i], C_t.columns[j+3]]).values else 0
                X_il_t = np.insert(F_hat_t[i, :], 0, C_t_pre_i).reshape(-1, 1)
                C_t.iloc[i,j+3] = beta[:, j].T @ X_il_t

        # Update the main dataframe with the imputed values
        C = C[C['eom'] != dates[t]]
        C = pd.concat([C, C_t], ignore_index=True)
        C_t_pre = C_t
    return C






if __name__ == '__main__':
    
    fn = "data/kelly_data_without_nanocap.p"
    df = pd.read_pickle(fn)
    cols_to_drop = ["isin", "cusip", "sedol", "excntry"]
    df = df.drop(cols_to_drop, axis=1)
    df = df.dropna(subset=['ret_local_lead1m'])
    
    print ('======Finish Load Data=====')
    
    # date_1 = datetime.strptime('1962-02-28', '%Y-%m-%d').date()
    # date_2 = datetime.strptime('1962-03-31', '%Y-%m-%d').date()
    # df_sub = df[(df['eom'] == date_1) | (df['eom'] == date_2)]
    
    imputed = backward_cross_section_imputation(df)
    
    imputed.to_csv("data/factor_data_backwards_impute.csv", index=False)