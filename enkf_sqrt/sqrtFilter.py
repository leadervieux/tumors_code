import numpy as np
from scipy.linalg import svd

def sqrt_filter(ensemble, W, H, measurement, options=None):
    """
    Python Implementationof the Square Root Ensemble Kalman Filter (sqrtFilter.m).
    """
    if options is None: options = {}
    h_ones = options.get('Hones', 0)
    num_ens = ensemble.shape[1]
    
    # apriori mean and perturbations
    mean_f = np.mean(ensemble, axis=1, keepdims=True)
    LPfe = (ensemble - mean_f) / np.sqrt(num_ens - 1)
    
    # Measurement update
    if h_ones == 1:
        indices = H[:, 1].astype(int) - 1
        proj = LPfe[indices, :]
        innovation = measurement.reshape(-1, 1) - mean_f[indices]
    else:
        proj = H @ LPfe
        innovation = measurement.reshape(-1, 1) - H @ mean_f
            
    # Kalman Ke Gain
    # Ke = LPfe * proj' * inv(proj * proj' + W)
    innov_cov = proj @ proj.T + W
    Ke = LPfe @ proj.T @ np.linalg.inv(innov_cov)
    
    # Updated the mean (a priori mean + Kalman gain * innovation)
    mean_a = mean_f + Ke @ innovation

    # Update the perturbations
    Af = np.sqrt(num_ens - 1) * LPfe
    if h_ones == 1:
        Af_m = Af[indices, :]
    else:
        Af_m = H @ Af
        
    # Transformation matrix for the perturbations
    # HARHA = Af_m' * inv(W) * Af_m / (num_ens - 1)
    HARHA = (Af_m.T @ np.linalg.solve(W, Af_m)) / (num_ens - 1)
    
    # SVD for the numerical stability of the square root update
    U, S, Vh = svd(np.eye(num_ens) + HARHA)
    T_mat = U @ np.diag(1.0 / np.sqrt(S)) @ U.T
    
    # New ensemble
    Aa = Af @ T_mat
    updated_ensemble = mean_a + Aa
    
    return mean_a.flatten(), updated_ensemble