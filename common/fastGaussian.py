import numpy as np
from scipy.linalg import toeplitz, cholesky

def fast_gaussian(dimension, sdev, corr):
    """
    Python version of fastGaussian.m
    Generates a random vector following a Gaussian variogram in 2D.
    """
    # Dimension handling
    if isinstance(dimension, (int, float)):
        m = n = int(dimension)
    else:
        m, n = dimension
    
    mxn = m * n

    # If Sdev is a vector, we use a variance 1 and we multiply at the end.
    if np.size(sdev) > 1:
        variance = 1.0
        multiplier = 1.0
    else:
        variance = sdev
        multiplier = sdev

    # Correlation handling
    if isinstance(corr, (int, float, list, np.ndarray)):
        corr = np.atleast_1d(corr)
        if len(corr) == 1:
            corr = np.array([corr[0], corr[0]])
    
    # 1. Covariance matrix for the first dimension (m)
    dist = np.arange(m) / corr[0]
    T = np.exp(-(toeplitz(dist)**2)) + 1e-10 * np.eye(m) # La variance est appliquée à la fin
    # MATLAB: cholT' (Lower)
    cholT_L = cholesky(T, lower=True) 

    # 2. Covariance matrix for the second dimension (n)
    if corr[0] == corr[1] and n == m:
        cholT2_U = cholT_L.T
    else:
        dist2 = np.arange(n) / corr[1]
        T2 = np.exp(-(toeplitz(dist2)**2)) + 1e-10 * np.eye(n) # La variance est appliquée à la fin
        # MATLAB: cholT2 (Upper)
        cholT2_U = cholesky(T2, lower=False)
    
    # 3. Random vector generation and multiplication
    rand_x = np.random.randn(m, n)
    result = (multiplier * (cholT_L @ rand_x @ cholT2_U)).flatten(order='F')

    # If Sdev is a vector, we multiply the result by Sdev
    if np.size(sdev) > 1:
        if sdev.size == result.size:
            result = sdev.flatten() * result
        else:
            raise ValueError("fast_gaussian: Inconsistent dimension of Sdev")

    return result
