import numpy as np

def lambda_chem(b, lam0b, lam1b, xi_1):
    """
    Traduction de lambda_chem.m
    Calcule le potentiel de chimiotactisme basé sur une sigmoïde.
    """
    b_max = 0.3
    lamb = lam0b - lam1b / (1 + np.exp(-xi_1 * (b - b_max)))
    return lamb