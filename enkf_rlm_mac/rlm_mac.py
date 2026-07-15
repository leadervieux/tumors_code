"""
rlm_mac.py

Translation Python of RLM_MAC.m: Iterative Ensemble Smoother, Levenberg-Marquardt form,
"minimum-average-cost" variant (RLM-MAC).
Adapted to interface with:
    - the tumor simulator (fibroblasts / cancer cells) already translated to Python (three_phase_simulator_compartment);
    - the existing square-root filter (sqrtFilter.py) is NOT used here: RLM-MAC computes its own gain via SVD (see below), as in the original MATLAB.
Two injection points allow you to hook your code without duplicating logic:
    - forward_sim_func(ensemble) -> sim_data_full:
          reruns the simulator on ALL members of `ensemble` (nm, ne+1)
          and returns the non-normalized simulated data, line-aligned with `measurement` (nd, ).
    - bounds_func(ensemble) -> ensemble:
          applies physical bounds (positivity, vascular/growth masks, log(khat_w) saturation, etc.) on the updated ensemble, column by column.
"""

from __future__ import annotations

import os
import numpy as np
from scipy.io import savemat


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def sqrt_w(in_w):
    """
    Root square (if `in_w` is a vector of variances) or transposed Cholesky factor (if `in_w` is a covariance matrix).
    Used to convert a variance into a standard deviation / normalization factor.
    """
    in_w = np.asarray(in_w)
    if in_w.ndim == 1 or in_w.shape[1] == 1:
        return np.sqrt(in_w).reshape(-1)
    else:
        # chol(inW)' en MATLAB (upper) == cholesky "lower"
        return np.linalg.cholesky(in_w)


def normalize_data(data, s_w):
    """Normalize `data` (nd, n) by the standard deviation / sqrt(W) matrix `s_w`."""
    data = np.asarray(data, dtype=float)
    s_w = np.asarray(s_w)
    if s_w.ndim == 1:
        return data / s_w.reshape(-1, 1)
    else:
        return np.linalg.solve(s_w, data)


def get_data_mismatch(sim_data, W, measurement):
    """
    Calculate the data mismatch for the ensemble.

    Parameters
    ----------
    sim_data : (nd, ne) ndarray
        Simulated data, already normalized by the observation error stds.
    W : (nd,) ndarray
        Weights of the observation elements (== 1 after normalization).
    measurement : (nd,) ndarray
        Actual measurements, normalized.

    Returns
    -------
    obj : float
        Mean misfit over the ensemble.
    obj_std : float
        Standard deviation of the misfit over the ensemble.
    obj_real : (ne,) ndarray
        Misfit of each realization.
    mismatch_mtx : (nd, ne) ndarray
        Detailed misfit by data and by realization.
    """
    sim_data = np.asarray(sim_data, dtype=float)
    measurement = np.asarray(measurement, dtype=float).reshape(-1, 1)
    W = np.asarray(W, dtype=float).reshape(-1, 1)

    mismatch_mtx = ((sim_data - measurement) ** 2) / (W ** 2)
    obj_real = mismatch_mtx.sum(axis=0)
    obj = obj_real.mean()
    ne = sim_data.shape[1]
    obj_std = obj_real.std(ddof=1) if ne > 1 else 0.0

    return obj, obj_std, obj_real, mismatch_mtx


def default_kalman_options(kalman_options: dict) -> dict:
    """
    Complete `kalman_options` with reasonable default values (lightweight equivalent of defaultKalmanOptions.m, not provided here).
    """
    defaults = {
        "append_mean": True,
        "maxIter": 8,
        "maxInnerIter": 5,
        "beta": 1.0,               
        "lambda": 1.0,              
        "lambda_increment_factor": 4.0,
        "lambda_reduction_factor": 0.5,
        "minReduction": 1.0,        
        "tsvdData": 0.99,
        "ignoreUninformativeMeasurements": False,
    }
    out = dict(defaults)
    out.update(kalman_options or {})
    return out


# ----------------------------------------------------------------------
# Principal function: RLM-MAC
# ----------------------------------------------------------------------

def rlm_mac(iter_, ensemble, sim_data, W, Wbase, measurement,
            kalman_options, dir_path, forward_sim_func,
            bounds_func=None, rng=None,
            debug_filename="debugRLM_MAC.txt"):
    """
    Iterative ensemble smoother with minimum-average-cost (RLM-MAC).

    Parameters
    ----------
    iter_ : int
        Index of the starting iteration (0 for a cold start).
    ensemble : (nm, ne+1) ndarray
        Set of static parameters. The last column MUST be the
        ensemble mean (kalman_options['append_mean'] = True).
    sim_data : (nd, ne+1) ndarray
        Simulated data (NOT normalized) associated with `ensemble`.
    W : (nd,) ndarray
        Vector of data weights (usually 1 after
        normalization by Wbase).
    Wbase : (nd,) or (nd, nd) ndarray
        Variance (vector) or covariance (matrix) of the observation errors.
    measurement : (nd,) ndarray
        Real measurements, NOT normalized (normalization is done internally,
        as in runIES.m original -> here we do it in rlm_mac to remain autonomous;
        see run_ies.py for the call).
    kalman_options : dict
        See default_kalman_options().
    dir_path : str
        Working directory (ensemble#.mat files, etc.).
    forward_sim_func : callable
        forward_sim_func(ensemble) -> sim_data (nd, ne+1) NOT normalized,
        relance le simulateur sur l'ensemble complet (ne membres + moyenne).
    bounds_func : callable, optional
        bounds_func(ensemble) -> ensemble, apply physical condition after each update
    rng : numpy.random.Generator, optional
        Random generator.

    Returns
    --------
    ensemble : (nm, ne+1) ndarray
        Final ensemble.
    sim_data : (nd, ne+1) ndarray
        Last simulated and normalized data.
    iter_ : int
        Last iteration index.
    """
    kalman_options = default_kalman_options(kalman_options)
    if not kalman_options.get("append_mean", False):
        raise ValueError("rlm_mac: kalman_options['append_mean'] doit être True.")

    if rng is None:
        rng = np.random.default_rng()

    os.makedirs(dir_path, exist_ok=True)
    debug_mode = "w" if iter_ == 0 else "a"
    debug_path = os.path.join(dir_path, debug_filename)
    

    W = np.asarray(W, dtype=float).reshape(-1)
    measurement = np.asarray(measurement, dtype=float).reshape(-1)
    ensemble = np.array(ensemble, dtype=float, copy=True)
    sim_data = np.array(sim_data, dtype=float, copy=True)

    if sim_data.shape[0] != measurement.shape[0]:
        raise ValueError("rlm_mac: inconsistents sim_data and measurement dimensions.")
    if W.shape[0] != measurement.shape[0]:
        raise ValueError("rlm_mac: inconsistents W and measurement dimensions.")

    ne = ensemble.shape[1] - 1
    nd = W.shape[0]

    s_wbase = sqrt_w(Wbase)

    with open(debug_path, debug_mode) as fid:
        fid.write(f"iteration number {iter_}\n")

        # perturbation of data measurement
        perturbed_data = measurement.reshape(-1, 1) + W.reshape(-1, 1) * rng.standard_normal((nd, ne))

        sim_data = normalize_data(sim_data, s_wbase)

        lambda_ = kalman_options["lambda"]
        if iter_ > 0:
            ens_file = os.path.join(dir_path, f"ensemble{iter_}.mat")
            if os.path.exists(ens_file):
                from scipy.io import loadmat
                lambda_ = float(loadmat(ens_file)["lambda"].squeeze())

        obj, obj_std, obj_real, _ = get_data_mismatch(sim_data[:, :ne], W, measurement)
        print(f"obj={obj:.6g} objStd={obj_std:.6g}")
        fid.write("rlm-mac  iter\titerLambda\tobjNew\tobjStdNew\tlambda\tchangeM\n\n")
        fid.write(f"rlm-mac  {iter_}\t0\t\t{obj:.6f}\t{obj_std:.6f}\n\n")
        savemat(os.path.join(dir_path, f"objRealIter{iter_}.mat"), {"objReal": obj_real})

        small_reduction = False
        dm_threshold = (kalman_options["beta"] ** 2) * len(measurement)

        # ---------------- outer loop ----------------
        while iter_ < kalman_options["maxIter"] and obj > dm_threshold:

            fid.write(f"number of data is {sim_data.shape[0]}\n")
            print("--------------------------------------------")
            print(f"-- Iteration step: {iter_} --")
            print(f"number of data is {sim_data.shape[0]}")

            sim_mean = sim_data[:, -1]
            delta_d = sim_data[:, :ne] - sim_mean[:, None]

            sim_data_upd = sim_data
            perturbed_data_eff = perturbed_data
            if kalman_options.get("ignoreUninformativeMeasurements", False):
                var_test = delta_d.var(axis=1, ddof=0)
                keep = np.flatnonzero(var_test != 0)
                delta_d = delta_d[keep, :]
                sim_data_upd = sim_data[keep, :]
                perturbed_data_eff = perturbed_data[keep, :]

                print(f"number of data for updating is {sim_data_upd.shape[0]}")
                fid.write(f"number of data for updating is {sim_data_upd.shape[0]}\n")

                if keep.size == 0:
                    print("WARNING rlm_mac: no variability in ensemble prediction")
                    fid.write("WARNING: no variability in ensemble prediction\n")
                    return ensemble, sim_data, iter_

            # SVD de deltaD (troncature selon tsvdData)
            Ud, Wd_full, Vt = np.linalg.svd(delta_d, full_matrices=False)
            Vd_full = Vt.T
            total = Wd_full.sum()
            cum = np.cumsum(Wd_full)
            svd_pd = int(np.searchsorted(cum / total, kalman_options["tsvdData"]) + 1)
            svd_pd = min(svd_pd, Wd_full.size)
            print(f"svdPd={svd_pd}")
            fid.write(f"number of singular value retained {svd_pd}\n")

            Ud = Ud[:, :svd_pd]
            Vd = Vd_full[:, :svd_pd]
            Wd = Wd_full[:svd_pd]

            iter_lambda = 1
            mean_en = ensemble[:, -1]  # ensemble mean before update
            delta_m = ensemble[:, :ne] - mean_en[:, None]

            # ---------------- inner loop / lambda ----------------
            while iter_lambda < kalman_options["maxInnerIter"]:

                alpha = lambda_ * np.sum(Wd ** 2) / svd_pd
                alpha = max(alpha, 1e-3)  

                x1 = Vd * (Wd / (alpha + Wd ** 2))  # (ne, svd_pd), broadcasting

                X = sim_data_upd[:, :ne] - perturbed_data_eff
                increment = delta_m @ x1 @ (Ud.T @ X)

                ensemble_old = ensemble.copy()
                ensemble[:, :ne] = ensemble[:, :ne] - increment
                ensemble[:, ne] = ensemble[:, :ne].mean(axis=1)  # append mean

                if bounds_func is not None:
                    ensemble = bounds_func(ensemble)
                    ensemble[:, ne] = ensemble[:, :ne].mean(axis=1)  # re-append

                change_m = np.sqrt(np.sum((ensemble[:, -1] - ensemble_old[:, -1]) ** 2))
                print(f"average change to model variable {change_m:.6g}")

                sim_data_old = sim_data

                # ---- re-launching simulator ----
                sim_data_raw = forward_sim_func(ensemble)
                savemat(os.path.join(dir_path, "tmpSimData.mat"), {"simData": sim_data_raw})
                sim_data = normalize_data(sim_data_raw, s_wbase)

                obj_new, obj_std_new, obj_real, _ = get_data_mismatch(
                    sim_data[:, :ne], W, measurement)
                print(f"   objNew={obj_new:.6g} objStdNew={obj_std_new:.6g}")

                fid.write(
                    f"rlm-mac  {iter_}\t{iter_lambda}\t\t{obj_new:.6f}\t"
                    f"{obj_std_new:.6f}\t{lambda_:.6f}\t{change_m:.6f}\n\n"
                )

                if obj_new > obj:
                    lambda_ *= kalman_options["lambda_increment_factor"]
                    print(f"increasing Lambda to {lambda_:.6g}")
                    iter_lambda += 1
                    sim_data = sim_data_old
                    ensemble = ensemble_old
                else:
                    lambda_ *= kalman_options["lambda_reduction_factor"]
                    print(f"reducing Lambda to {lambda_:.6g}")

                    if abs(obj_new - obj) / abs(obj) * 100 < kalman_options["minReduction"]:
                        small_reduction = True

                    obj_std = obj_std_new
                    obj = obj_new
                    break

            else:
                # The inner loop is finished, no better update found
                lambda_ *= kalman_options["lambda_increment_factor"]
                if lambda_ < kalman_options["lambda"]:
                    lambda_ = kalman_options["lambda"]
                msg = "terminating iterations: iterLambda>=maxInnerIter"
                print(msg)
                fid.write(msg + "\n")

            # ---- Save end of external interation ----
            iter_ += 1
            savemat(os.path.join(dir_path, f"ensemble{iter_}.mat"),
                    {"ensemble": ensemble, "lambda": lambda_})
            tmp_path = os.path.join(dir_path, "tmpSimData.mat")
            final_path = os.path.join(dir_path, f"simulatedDataIter{iter_}.mat")
            if os.path.exists(tmp_path):
                os.replace(tmp_path, final_path)
            savemat(os.path.join(dir_path, f"objRealIter{iter_}.mat"), {"objReal": obj_real})

            if small_reduction:
                msg = (f"terminating iterations: reduction of objective function "
                       f"is less than {kalman_options['minReduction']}%")
                print(msg)
                fid.write(msg + "\n")
                break

        if iter_ >= kalman_options["maxIter"]:
            msg = "terminating iterations: iter>=kalmanOptions.maxIter"
            print(msg)
            fid.write(msg + "\n")

    return ensemble, sim_data, iter_
