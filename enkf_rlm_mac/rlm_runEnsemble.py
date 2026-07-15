import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.ndimage import rotate, gaussian_filter
from scipy.io import savemat, loadmat
import time
from concurrent.futures import ProcessPoolExecutor

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "common"))
sys.path.insert(0, THIS_DIR)

from paths import DATA_DIR, output_dir
OUTPUT_DIR = output_dir(THIS_DIR)

from fastGaussian import fast_gaussian
from three_phase_simulator_compartment import (
    three_phase_simulator_compartment_full,      # simulator A (half+full)
    three_phase_simulator_compartment,           # simulator B (rapide)
    three_phase_simulator_compartment_forecast,  # simulator C (complete forecast)
)

from rlm_mac import rlm_mac, default_kalman_options

from Plot_EnsemblePred_Iteration_0_HalfFull import plot_half_full
from Plot_True_simulation_Verify import plot_true_simulation_verify


# ----------------------------------------------------------------------
# Wrappers for secure simulation of ensemble members (to handle NaN and adaptivity)
# ----------------------------------------------------------------------

def simulate_member_safe(alpha_c, member_params):
    """
    Wrapper to handle NTime adaptivity (Inc_dt) as in MATLAB.
    Uses simulator A (full) : returns half + full.
    """
    nan_result = True
    inc_dt = 0
    while nan_result and inc_dt < 5: # Limite à 5 essais
        res_half, res_full, p_half, p_full = three_phase_simulator_compartment_full(alpha_c, member_params, inc_dt)

        if np.isnan(p_full).any():
            inc_dt += 1
            print(f"Instability detected (A). Relaunching with Inc_dt={inc_dt}")
        else:
            nan_result = False
    return res_half, res_full, p_half, p_full


def simulate_member_B_safe(alpha_c, member_params):
    """
    Wrapper to handle NTime adaptivity (Inc_dt) as in MATLAB for simulator B (fast).
    Returns only half (measured) and IFP.
    """
    nan_result = True
    inc_dt = 0
    while nan_result and inc_dt < 5:
        measured_values, ifp = three_phase_simulator_compartment(alpha_c, member_params, inc_dt)

        if np.isnan(ifp).any():
            inc_dt += 1
            print(f"Instability detected (B). Relaunching with Inc_dt={inc_dt}")
        else:
            nan_result = False
    return measured_values, ifp


# ----------------------------------------------------------------------
# Fonctions de Rappel (Callbacks) requises par RLM-MAC
# ----------------------------------------------------------------------

def make_bounds_func(pdim, vasc_mask, growth_mask, log_khat_w_min=0.0, log_khat_w_max=5.0):
    """Generate the function that applies physical constraints in log space."""
    vasc_mask = np.asarray(vasc_mask, dtype=float).reshape(-1)
    growth_mask = np.asarray(growth_mask, dtype=float).reshape(-1)

    def bounds_func(ensemble):
        ensemble = np.array(ensemble, copy=True)
        # alpF >= 0
        ensemble[0:pdim, :] = np.maximum(ensemble[0:pdim, :], 0.0)
        ensemble[pdim:2 * pdim, :] = np.clip(ensemble[pdim:2 * pdim, :], log_khat_w_min, log_khat_w_max)
        ensemble[2 * pdim:3 * pdim, :] *= vasc_mask[:, None]
        ensemble[4 * pdim:5 * pdim, :] *= growth_mask[:, None]
        ensemble[5 * pdim:6 * pdim, :] *= growth_mask[:, None]
        return ensemble

    return bounds_func


def make_forward_sim_func(alpha_c_with_unc, pdim, meas_index, simulator="B", workers=None):
    """
    Generate the function that re-exponentiates khat_w and launches the simulator 
    on the ensemble to return predictions at the measurement points.
    """
    if workers is None:
        workers = max(1, os.cpu_count() // 2)
    sim_func = simulate_member_safe if simulator == "A" else simulate_member_B_safe

    def forward_sim_func(ensemble):
        n_members = ensemble.shape[1]
        ensemble_direct = ensemble.copy()
        ensemble_direct[pdim:2 * pdim, :] = np.exp(ensemble_direct[pdim:2 * pdim, :])

        sim_full = np.zeros((pdim, n_members))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(sim_func, alpha_c_with_unc, ensemble_direct[:, k])
                       for k in range(n_members)]
            for k, future in enumerate(futures):
                result = future.result()
                measured_values = result[0] 
                sim_full[:, k] = measured_values
        return sim_full[meas_index, :]

    return forward_sim_func


# ----------------------------------------------------------------------
# Principal function to run the ensemble simulation and RLM-MAC assimilation
# ----------------------------------------------------------------------

def run_ensemble_simulation():

    dim = (61,61)
    Ny, Nx = dim
    pdim = Nx * Ny

    dx = 1.0 / Nx
    dy = 1.0 / Ny
    X, Y = np.meshgrid(0 + dx * 0.5 + np.arange(Nx) * dx, 0 + dy * 0.5 + np.arange(Ny) * dy)
    
    rho = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)

    # Definition of vascular and lymphatic positions
    vascPos = np.ones(dim)
    lymphPos = np.ones(dim)
    blankLymphPos = (X - 0.5)**2 + (Y - 0.5)**2 < 0.25**2
    lymphPos = lymphPos - blankLymphPos.astype(float)

    print("Step A : basic data")

    # --- 2. Ensemble Configuration ---
    data = loadmat(os.path.join(DATA_DIR, 'initial_cancer_cells.mat'))
    alpha_c_with_unc = data['alpha_c_with_unc']

    print("Step B : generate initial ensemble")

    num_ens = 100
    initial_ensemble = np.zeros((6 * pdim, num_ens))

    # 3.1 Initialization of Fibroblasts (alpF)
    fib_pos = np.exp(-200 * (rho.flatten(order='F') - 0.175)**2)
    for k in range(num_ens):
        noise = fast_gaussian(dim, 0.075, 4)
        alp_f = 0.125 + noise
        alp_f[alp_f < 0] = 0
        initial_ensemble[0:pdim, k] = alp_f * fib_pos

    # 3.2 Generation of band structures (khat_w)
    tumor_region = (alpha_c_with_unc > 0.01).flatten(order='F')
    for k in range(num_ens):
        C_accum = np.zeros(dim)
        for line_idx in range(4):
            B = np.zeros(dim)
            gauss_line = np.zeros(dim)
            start_col = 30 + int(np.round(np.random.rand() * 20 - 10))
            B[0, start_col] = 1
            gauss_line[0:2, start_col-2:start_col+3] = (1 + fast_gaussian((2, 5), 0.2, [1, 1])).reshape(2, 5, order='F')
            
            curr_r, curr_c = 0, start_col
            while curr_r < Ny - 2:
                curr_r += 1
                rand_val = np.random.rand()
                if rand_val < 1/3:
                    curr_c = min(Nx-1, curr_c + 1)
                elif rand_val > 2/3:
                    curr_c = max(0, curr_c - 1)
                B[curr_r, curr_c] = 1

                patch = (1 + fast_gaussian((2, 5), 0.2, [1, 1])).reshape(2, 5, order='F')
                row_lo, row_hi = curr_r, curr_r + 2
                col_lo, col_hi = curr_c - 2, curr_c + 3

                p_row_lo = max(0, -row_lo)
                p_row_hi = 2 - max(0, row_hi - Ny)
                p_col_lo = max(0, -col_lo)
                p_col_hi = 5 - max(0, col_hi - Nx)

                g_row_lo, g_row_hi = max(0, row_lo), min(Ny, row_hi)
                g_col_lo, g_col_hi = max(0, col_lo), min(Nx, col_hi)

                gauss_line[g_row_lo:g_row_hi, g_col_lo:g_col_hi] = patch[p_row_lo:p_row_hi, p_col_lo:p_col_hi]
            
            angle = (line_idx + 1) * 90
            rotated_B = rotate(B, angle, reshape=False, order=1)
            rotated_G = rotate(gauss_line, angle, reshape=False, order=1)
            C_accum += (rotated_B + rotated_G)
            
        C_final = gaussian_filter((C_accum.flatten(order='F') / 5.0 * tumor_region).reshape(dim, order='F') * 100 + 1, sigma=0.75)
        
        # KEY MODIFICATION: khat_w is stored in LOG for RLM-MAC!
        khat_w_direct = np.maximum(C_final.flatten(order='F'), 1e-6)  # Avoid log(0)
        initial_ensemble[pdim:2*pdim, k] = np.log(khat_w_direct)

    # 3.3 Filtration of vascular regions (Tv)
    vasc_mask = (((X - 0.5)**2 + (Y - 0.5)**2 < 0.13**2) & 
                 ((X - 0.5)**2 + (Y - 0.5)**2 > 0.075**2)).flatten(order='F')
    
    for k in range(num_ens):
        tv = 0.0001 + np.random.rand() * (0.005 - 0.0001)
        initial_ensemble[2*pdim:3*pdim, k] = tv * vasc_mask
        
        tl_field = np.zeros(dim)
        tl_field[(X - 0.5)**2 + (Y - 0.5)**2 > 0.4**2] = 0.0055
        initial_ensemble[3*pdim:4*pdim, k] = tl_field.flatten(order='F')

    # 3.4 Growth regions (Tg and Tg2)
    growth_mask = ((X - 0.5)**2 + (Y - 0.5)**2 < 0.2**2).flatten(order='F')
    for k in range(num_ens):
        initial_ensemble[4*pdim:5*pdim, k] = 0.0 * growth_mask
        initial_ensemble[5*pdim:6*pdim, k] = 0.0 * growth_mask

    # KEY MODIFICATION: Adding the ensemble mean as the last member (required by RLM-MAC)
    ensemble = np.hstack([initial_ensemble, initial_ensemble.mean(axis=1, keepdims=True)])

    print(f"Shape of RLM-MAC ensemble (with added mean): {ensemble.shape}")
    print("Step C: Initial ensemble generated")

    # --- 4. Loading Truth ---
    try:
        true_init = loadmat(os.path.join(DATA_DIR, 'Ensemble_Initial_E20_True_April14_theta0_K0.mat'))
        true_sol = loadmat(os.path.join(DATA_DIR, 'Ensemble_Solution_E20_True_Growth_May08_vary_theta0_K0_reduced.mat'))

        initial_ensemble_true = true_init['initialEnsemble_True']
        k_true = 6
        initial_true_model = initial_ensemble_true[:, k_true]
        
        true_alpha_c_half = true_sol['alpha_c_Half_True'][:, k_true]
        true_ifp_half = true_sol['IFP_Half_True'][:, k_true]

        plot_true_simulation_verify(true_alpha_c_half, true_ifp_half,
                                    true_sol['alpha_c_Full_True'][:, k_true], true_sol['IFP_Full_True'][:, k_true],
                                    initial_true_model, X, Y, alpha_c_with_unc)
        print("Step D: Finished (Truth visualized)")
    except FileNotFoundError:
        print("Step D: Files not found, skipping visual truth verification.")


    # --- 5. Ensemble simulation (First prediction) ---
    ens_pred_half = np.zeros((pdim, num_ens + 1))
    ens_pred_full = np.zeros((pdim, num_ens + 1))
    ens_ifp_half = np.zeros((pdim, num_ens + 1))
    ens_ifp_full = np.zeros((pdim, num_ens + 1))

    start_time = time.time()
    workers = max(1, os.cpu_count() // 2)
    print(f"Starting simulation of {num_ens + 1} members (including the mean) ({workers} workers)...")
    
    # For MATLAB saving, we pass directly (linearly) temporarily
    ensemble_direct = ensemble.copy()
    ensemble_direct[pdim:2*pdim, :] = np.exp(ensemble_direct[pdim:2*pdim, :])
    savemat(os.path.join(OUTPUT_DIR, 'initialEnsemble_Python_rlm.mat'), {
        'initialEnsemble': np.array(ensemble_direct[:, :num_ens], order='F'),
        'alpha_c_with_unc': np.array(alpha_c_with_unc, order='F')
    }) 
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(simulate_member_safe, alpha_c_with_unc, ensemble_direct[:, k].copy()) for k in range(num_ens + 1)]
        for k, future in enumerate(futures):
            res_half, res_full, p_half, p_full = future.result()
            ens_pred_half[:, k] = res_half
            ens_pred_full[:, k] = res_full
            ens_ifp_half[:, k] = p_half
            ens_ifp_full[:, k] = p_full
            if k < num_ens and (k+1) % 10 == 0:
                print(f"Terminé : {k+1}/{num_ens}")
                plot_half_full(alpha_c_with_unc, ensemble_direct[:, k], res_half, res_full, p_half, p_full, X, Y)

    print(f"First prediction completed in {time.time() - start_time:.2f}s")

    # --- 5.1 Saving the first prediction (without the mean column) ---
    init_pred_results = {
        'ensembleOfPredictedObservations': np.array(ens_pred_half[:, :num_ens], order='F'),
        'ensembleOfIFPs': np.array(ens_ifp_half[:, :num_ens], order='F'),
        'ensembleOfPredictedObservationsFull': np.array(ens_pred_full[:, :num_ens], order='F'),
        'ensembleOfIFPsFull': np.array(ens_ifp_full[:, :num_ens], order='F'),
        'alpha_c_with_unc': alpha_c_with_unc,
        'initialEnsemble': ensemble_direct[:, :num_ens]
    }
    savemat(os.path.join(OUTPUT_DIR, 'rlm_numEns100FullAddNoiseLowPl_Initial_Python.mat'), init_pred_results)
    print("File for test.py generated : numEns100FullAddNoiseLowPl_Initial_Python.mat")

    print("End of the first prediction, starting RLM-MAC assimilation.")


    # --- 6. Configuration and launching of RLM-MAC (replaces EnKF) ---
    num_iterations = 4
    add_unc = 0.05

    # Loading measurements (from MATLAB file)
    meas = loadmat(os.path.join(DATA_DIR, "measCancerCells.mat"))
    measurement_full = meas['measurement'].flatten(order='F')

    # Selection of the most informative measurements based on variance across ensemble predictions
    variances = np.var(ens_pred_half[:, :num_ens], axis=1)
    meas_index = np.argsort(variances)[-250:] # Top 250
    measurement = measurement_full[meas_index]

    # Weighting matrix for RLM-MAC
    nd = len(meas_index)
    Wbase = (add_unc ** 2) * np.ones(nd)
    W = np.ones(nd)

    # RLM-MAC options
    kalman_options = default_kalman_options({
        "maxIter": num_iterations,
        "beta": 0.0,  # Pas d'arrêt précoce pour forcer à faire "num_iterations"
        "maxInnerIter": 5,
        "lambda": 1.0,
        "lambda_increment_factor": 4.0,
        "lambda_reduction_factor": 0.5,
        "minReduction": 0.5,
        "tsvdData": 0.99,
        "ignoreUninformativeMeasurements": True,
    })

    # Creating the bounds and forward simulation functions for RLM-MAC
    bounds_func = make_bounds_func(pdim, vasc_mask, growth_mask)
    forward_sim_func = make_forward_sim_func(alpha_c_with_unc, pdim, meas_index, simulator="B", workers=workers)

    # sim_data must contain the initial simulated observations for all members (including the mean)
    sim_data = ens_pred_half[meas_index, :]

    dir_path = os.path.join(OUTPUT_DIR, "rlm_mac_run/")
    os.makedirs(dir_path, exist_ok=True)

    print("Starting RLM-MAC...")
    ensemble_updated, sim_data_updated, last_iter = rlm_mac(
        iter_ = 0,
        ensemble = ensemble,
        sim_data = sim_data,
        W = W,
        Wbase = Wbase,
        measurement = measurement,
        kalman_options = kalman_options,
        dir_path = dir_path,
        forward_sim_func = forward_sim_func,
        bounds_func = bounds_func
    )
    print(f"RLM-MAC successfully completed at iteration {last_iter}.")


    # --- 7. Final saving (Linearization) ---
    
    current_ensemble = ensemble_updated[:, :num_ens].copy()
    current_ensemble[pdim:2*pdim, :] = np.exp(current_ensemble[pdim:2*pdim, :])

    mean_state = ensemble_updated[:, -1].copy()
    mean_state[pdim:2*pdim] = np.exp(mean_state[pdim:2*pdim])

    # Final re-simulation of updated members using simulator B
    ens_pred_half_final = np.zeros((pdim, num_ens))
    ens_ifp_half_final = np.zeros((pdim, num_ens))
    print("Final re-simulation of updated members (simulator B)...")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(simulate_member_B_safe, alpha_c_with_unc, current_ensemble[:, k]) for k in range(num_ens)]
        for k, future in enumerate(futures):
            measured_values, ifp = future.result()
            ens_pred_half_final[:, k] = measured_values
            ens_ifp_half_final[:, k] = ifp

    results = {
        'alpha_c_with_unc': alpha_c_with_unc,
        'ensembleOfPredictedObservations': ens_pred_half_final,
        'initialEnsemble': ensemble_direct[:, :num_ens],
        'ensembleOfIFPs': ens_ifp_half_final,
        'ensembleOfIFPsFull': ens_ifp_full[:, :num_ens], # Inchangé
        'ensembleOfPredictedObservationsFull': ens_pred_full[:, :num_ens], # Inchangé
        'updatedEnsemble': current_ensemble,
        'measIndex': meas_index + 1, # Indexation 1-based pour MATLAB
        'meanstate': mean_state,
        'X': X, 'Y': Y
    }
    savemat(os.path.join(OUTPUT_DIR, 'rlm_PlLowfinalResAddNoise_Python_Final.mat'), results)
    print("Final results saved.")


    # --- 8. Forecasting ---
    print("Forecast based on UPDATED mean STARTED")
    (alpha_c_half_upd, alpha_c_full_upd, ifp_half_upd, ifp_full_upd,
     alpha_f_half_upd, alpha_f_full_upd,
     uu_w_x_half_upd, uu_w_y_half_upd,
     uu_w_x_full_upd, uu_w_y_full_upd) = three_phase_simulator_compartment_forecast(
        alpha_c_with_unc, mean_state)
    print("Forecast based on UPDATED mean FINISHED")

    savemat(os.path.join(OUTPUT_DIR, 'rlm_numEns100IterationsForecastAddNoisePlLow_upd_mean_Python.mat'), {
        'alpha_c_Half_upd_mean': alpha_c_half_upd,
        'alpha_c_Full_upd_mean': alpha_c_full_upd,
        'IFP_Half_upd_mean': ifp_half_upd,
        'IFP_Full_upd_mean': ifp_full_upd,
        'alpha_f_Half_upd_mean': alpha_f_half_upd,
        'alpha_f_Full_upd_mean': alpha_f_full_upd,
        'uu_W_x_Half_upd_mean': uu_w_x_half_upd,
        'uu_W_y_Half_upd_mean': uu_w_y_half_upd,
        'uu_W_x_Full_upd_mean': uu_w_x_full_upd,
        'uu_W_y_Full_upd_mean': uu_w_y_full_upd,
    })

    # --- 8.1 Forecast based on the initial mean (WITHOUT updating) ---
    print("Forecast based on mean without updating STARTED")
    initial_mean_params = ensemble_direct[:, :num_ens].mean(axis=1) # Déjà en linéaire
    (alpha_c_half_mean, alpha_c_full_mean, ifp_half_mean, ifp_full_mean,
     alpha_f_half_mean, alpha_f_full_mean,
     uu_w_x_half_mean, uu_w_y_half_mean,
     uu_w_x_full_mean, uu_w_y_full_mean) = three_phase_simulator_compartment_forecast(
        alpha_c_with_unc, initial_mean_params)
    print("Forecast based on mean without updating FINISHED")

    savemat(os.path.join(OUTPUT_DIR, 'rlm_numEns100ForecastAddNoisePlLow_mean_Python.mat'), {
        'alpha_c_Half_mean': alpha_c_half_mean,
        'alpha_c_Full_mean': alpha_c_full_mean,
        'IFP_Half_mean': ifp_half_mean,
        'IFP_Full_mean': ifp_full_mean,
        'alpha_f_Half_mean': alpha_f_half_mean,
        'alpha_f_Full_mean': alpha_f_full_mean,
        'uu_W_x_Half_mean': uu_w_x_half_mean,
        'uu_W_y_Half_mean': uu_w_y_half_mean,
        'uu_W_x_Full_mean': uu_w_x_full_mean,
        'uu_W_y_Full_mean': uu_w_y_full_mean,
    })
    print("Forecasts completed. Results saved.")

if __name__ == "__main__":
    run_ensemble_simulation()