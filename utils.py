import math
import numpy as np

def normalize_psd(PSD,delta_f):
    total_power = np.sum(PSD)*delta_f
    return PSD/total_power


def MAC_matrix(eigenmatrix1:np.ndarray,eigenmatrix2:np.ndarray=None):
    """Modal Assurance Criterion Matrix Calculation

    Args:
        eigenmatrix1 (np.ndarray): Mode Shape matrix number 1 N x R, N degrees of freedom, R modes

        eigenmatrix2 (np.ndarray, optional): Mode Shape matrix number 2 N x R Defaults to eigenmatrix1 for Auto MAC calculation.

    """
    if eigenmatrix2 is None:
        eigenmatrix2=eigenmatrix1

    Nmodes = eigenmatrix1.shape[1]

    MAC = np.zeros((Nmodes,Nmodes))

    for r in range(Nmodes):
        for q in range(Nmodes):
            MAC[r,q] =(np.abs(eigenmatrix1[:,r].T @ eigenmatrix2[:,q] )**2)/((eigenmatrix1[:,r].T @ eigenmatrix1[:,r])*(eigenmatrix2[:,q].T @ eigenmatrix2[:,q]))


    return MAC

# modeshapes_2dof_easy = np.array([[1.,1.],[1.,-1.]])

# print(MAC_matrix(modeshapes_2dof_easy))