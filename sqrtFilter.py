import numpy as np
from scipy.linalg import svd

def sqrt_filter(ensemble, W, H, measurement, options=None):
    """
    Implémentation Python du Square Root Ensemble Kalman Filter (sqrtFilter.m).
    """
    if options is None: options = {}
    h_ones = options.get('Hones', 0)
    num_ens = ensemble.shape[1]
    
    # Moyenne apriori et anomalies
    mean_f = np.mean(ensemble, axis=1, keepdims=True)
    LPfe = (ensemble - mean_f) / np.sqrt(num_ens - 1)
    
    # Gestion des mesures (simplifiée ici, sans ignoreUninformative pour l'instant)
    if h_ones == 1:
        # H[:, 1] contient les indices MATLAB (1-based)
        indices = H[:, 1].astype(int) - 1
        proj = LPfe[indices, :]
        innovation = measurement.reshape(-1, 1) - mean_f[indices]
    else:
        proj = H @ LPfe
        innovation = measurement.reshape(-1, 1) - H @ mean_f
            
    # Gain de Kalman Ke
    # Ke = LPfe * proj' * inv(proj * proj' + W)
    innov_cov = proj @ proj.T + W
    Ke = LPfe @ proj.T @ np.linalg.inv(innov_cov)
    
    # Mise à jour de la moyenne (aposteriori)
    mean_a = mean_f + Ke @ innovation

    # Mise à jour de la covariance (Ensemble Transform)
    Af = np.sqrt(num_ens - 1) * LPfe
    if h_ones == 1:
        Af_m = Af[indices, :]
    else:
        Af_m = H @ Af
        
    # Calcul de la matrice de transformation T
    # HARHA = Af_m' * inv(W) * Af_m / (num_ens - 1)
    HARHA = (Af_m.T @ np.linalg.solve(W, Af_m)) / (num_ens - 1)
    
    # SVD pour la stabilité numérique
    U, S, Vh = svd(np.eye(num_ens) + HARHA)
    T_mat = U @ np.diag(1.0 / np.sqrt(S)) @ U.T
    
    # Nouvel ensemble
    Aa = Af @ T_mat
    updated_ensemble = mean_a + Aa
    
    return mean_a.flatten(), updated_ensemble