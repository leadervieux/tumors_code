import numpy as np

def lambda_TGF(c, lam0c, lam1c, xi_2):
    """
    Traduction de lambda_TGF.m
    Calcule le potentiel de réponse au TGF-beta basé sur une sigmoïde.
    """
    c_max = 0.2
    lamb = lam0c - lam1c / (1 + np.exp(-xi_2 * (c - c_max)))
    return lamb