import numpy as np

def lambda_chem(b, lam0b, lam1b, xi_1):
    """
    Translation from lambda_chem.m
    Calculates the chemotactic potential based on a sigmoid function.
    """
    b_max = 0.3
    lamb = lam0b - lam1b / (1 + np.exp(-xi_1 * (b - b_max)))
    return lamb