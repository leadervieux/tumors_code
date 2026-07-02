import numpy as np
from scipy.linalg import toeplitz, cholesky

def fast_gaussian(dimension, sdev, corr):
    """
    Version Python de fastGaussian.m
    Génère un vecteur aléatoire suivant un variogramme gaussien en 2D.
    """
    # Gestion des dimensions
    if isinstance(dimension, (int, float)):
        m = n = int(dimension)
    else:
        m, n = dimension
    
    mxn = m * n

    # Si Sdev est un vecteur, on utilise une variance de 1 et on multiplie à la fin
    if np.size(sdev) > 1:
        variance = 1.0
        multiplier = 1.0
    else:
        variance = sdev
        multiplier = sdev

    # Gestion de la corrélation
    if isinstance(corr, (int, float, list, np.ndarray)):
        corr = np.atleast_1d(corr)
        if len(corr) == 1:
            corr = np.array([corr[0], corr[0]])
    
    # 1. Matrice de covariance pour la première dimension (m)
    dist = np.arange(m) / corr[0]
    T = np.exp(-(toeplitz(dist)**2)) + 1e-10 * np.eye(m) # La variance est appliquée à la fin
    # MATLAB: cholT' (Lower)
    cholT_L = cholesky(T, lower=True) 

    # 2. Matrice de covariance pour la deuxième dimension (n)
    if corr[0] == corr[1] and n == m:
        cholT2_U = cholT_L.T
    else:
        dist2 = np.arange(n) / corr[1]
        T2 = np.exp(-(toeplitz(dist2)**2)) + 1e-10 * np.eye(n) # La variance est appliquée à la fin
        # MATLAB: cholT2 (Upper)
        cholT2_U = cholesky(T2, lower=False)
    
    # 3. Tirage aléatoire et ajustement de la covariance
    rand_x = np.random.randn(m, n)
    # On applique le multiplicateur (Sdev) à la fin pour garantir l'écart-type
    result = (multiplier * (cholT_L @ rand_x @ cholT2_U)).flatten(order='F')

    # Application de la pondération spatiale si Sdev est un vecteur
    if np.size(sdev) > 1:
        if sdev.size == result.size:
            result = sdev.flatten() * result
        else:
            raise ValueError("fast_gaussian: Inconsistent dimension of Sdev")

    return result
