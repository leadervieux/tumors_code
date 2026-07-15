import numpy as np

def lambda_TGF(c, lam0c, lam1c, xi_2):
    """
    Translation from lambda_TGF.m
    Calculates the TGF-beta response potential based on a sigmoid function.
    """
    c_max = 0.2
    lamb = lam0c - lam1c / (1 + np.exp(-xi_2 * (c - c_max)))
    return lamb