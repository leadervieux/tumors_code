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
    three_phase_simulator_compartment,           # simulator B (rapide, re-simulation EnKF)
    three_phase_simulator_compartment_forecast,  # simulator C (forecast complet)
)
# On s'assure que le nom du fichier du filtre est correct (enkf.py ou sqrtFilter.py)
from sqrtFilter import sqrt_filter
from Plot_EnsemblePred_Iteration_0_HalfFull import plot_half_full
from Plot_True_simulation_Verify import plot_true_simulation_verify

def simulate_member_safe(alpha_c, member_params):
    """
    Wrapper to safely call the three_phase_simulator_compartment_full function.
    If NaN values are detected in the output, it retries with an increased inc_dt parameter.
    """
    nan_result = True
    inc_dt = 0
    while nan_result and inc_dt < 5: # Limite à 5 essais pour éviter les boucles infinies
        res_half, res_full, p_half, p_full = three_phase_simulator_compartment_full(alpha_c, member_params, inc_dt)

        if np.isnan(p_full).any():
            inc_dt += 1
            print(f"Instability detected (A). Retry with Inc_dt={inc_dt}")
        else:
            nan_result = False
    return res_half, res_full, p_half, p_full


def simulate_member_B_safe(alpha_c, member_params):
    """
    Wrapper for the three_phase_simulator_compartment function (simulateur B).
    If NaN values are detected in the output, it retries with an increased inc_dt parameter
    """
    nan_result = True
    inc_dt = 0
    while nan_result and inc_dt < 5:
        measured_values, ifp = three_phase_simulator_compartment(alpha_c, member_params, inc_dt)

        if np.isnan(ifp).any():
            inc_dt += 1
            print(f"Instability detected (B). Retry with Inc_dt={inc_dt}")
        else:
            nan_result = False
    return measured_values, ifp

def run_ensemble_simulation():

    dim = (61,61)
    Ny, Nx = dim
    pdim = Nx * Ny

    dx = 1.0 / Nx
    dy = 1.0 / Ny
    x = 0 + dx * 0.5 + np.arange(Nx) * dx
    y = 0 + dy * 0.5 + np.arange(Ny) * dy
    X, Y = np.meshgrid(x, y)
    
    rho = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)

    # Definition of vascular and lymphatic masks
    vascPos = np.ones(dim)
    lymphPos = np.ones(dim)
    blankLymphPos = (X - 0.5)**2 + (Y - 0.5)**2 < 0.25**2
    lymphPos = lymphPos - blankLymphPos.astype(float)

    print("Step A : basic data")

    # --- 2. Ensemble configuration ---
    # The state vetor consists of 6 fields: [alpF, khat_w, T_vess, T_lymp, Kgrow, Thetagrow]
    data = loadmat(os.path.join(DATA_DIR, 'initial_cancer_cells.mat'))

    alpha_c_with_unc = data['alpha_c_with_unc']

    print("Step B : generate initial ensemble")

    num_ens = 100
    initial_ensemble = np.zeros((6 * pdim, num_ens))

    # 3.1 Fibroblastes Initialisation (alpF)
    fib_pos = np.exp(-200 * (rho.flatten(order='F') - 0.175)**2)
    for k in range(num_ens):
        noise = fast_gaussian(dim, 0.075, 4)
        alp_f = 0.125 + noise
        alp_f[alp_f < 0] = 0
        initial_ensemble[0:pdim, k] = alp_f * fib_pos

    # 3.2 Generation of band structures (khat_w / log conductivity)
    # Note: This part is simplified here for the demo, it follows the iterative logic of the MATLAB
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

                gauss_line[g_row_lo:g_row_hi, g_col_lo:g_col_hi] = \
                    patch[p_row_lo:p_row_hi, p_col_lo:p_col_hi]
            
            angle = (line_idx + 1) * 90
            rotated_B = rotate(B, angle, reshape=False, order=1) # order 1 = bilinear
            rotated_G = rotate(gauss_line, angle, reshape=False, order=1)
            C_accum += (rotated_B + rotated_G)
            
        C_final = gaussian_filter((C_accum.flatten(order='F') / 5.0 * tumor_region).reshape(dim, order='F') * 100 + 1, sigma=0.75)
        initial_ensemble[pdim:2*pdim, k] = C_final.flatten(order='F')

    # 3.3 Filtration parameters (Tvess, Tlymp)
    vasc_mask = (((X - 0.5)**2 + (Y - 0.5)**2 < 0.13**2) & 
                 ((X - 0.5)**2 + (Y - 0.5)**2 > 0.075**2)).flatten(order='F')
    
    for k in range(num_ens):
        tv = 0.0001 + np.random.rand() * (0.005 - 0.0001)
        initial_ensemble[2*pdim:3*pdim, k] = tv * vasc_mask
        
        tl_field = np.zeros(dim)
        tl_field[(X - 0.5)**2 + (Y - 0.5)**2 > 0.4**2] = 0.0055
        initial_ensemble[3*pdim:4*pdim, k] = tl_field.flatten(order='F')

    # 3.4 Growth parameters (Kgrow, Thetagrow)
    growth_mask = ((X - 0.5)**2 + (Y - 0.5)**2 < 0.2**2).flatten(order='F')
    for k in range(num_ens):
        initial_ensemble[4*pdim:5*pdim, k] = 00.0 * growth_mask
        initial_ensemble[5*pdim:6*pdim, k] = 0.0 * growth_mask

    num_ens = initial_ensemble.shape[1]

    print(f"Shape of initial_ensemble : {initial_ensemble.shape}")
    print(f"Shape of alpha_c_with_unc : {alpha_c_with_unc.shape}")

    print("Step C : Initial ensemble generated")

    # --- 4. Step D: run true model / Loading Truth ---
    initial_true_model = None
    true_model_data_half = None
    try:
        true_init = loadmat(os.path.join(DATA_DIR, 'Ensemble_Initial_E20_True_April14_theta0_K0.mat'))
        true_sol = loadmat(os.path.join(DATA_DIR, 'Ensemble_Solution_E20_True_Growth_May08_vary_theta0_K0_reduced.mat'))

        initial_ensemble_true = true_init['initialEnsemble_True']
        # member index for the true model (k_true) is set to 6, as in the MATLAB code
        k_true = 6
        initial_true_model = initial_ensemble_true[:, k_true]
        
        true_alpha_c_half = true_sol['alpha_c_Half_True'][:, k_true]
        true_ifp_half = true_sol['IFP_Half_True'][:, k_true]
        true_model_data_half = np.concatenate([true_alpha_c_half, true_ifp_half])

        plot_true_simulation_verify(true_alpha_c_half, true_ifp_half,
                                    true_sol['alpha_c_Full_True'][:, k_true], true_sol['IFP_Full_True'][:, k_true],
                                    initial_true_model, X, Y, alpha_c_with_unc)
        print("Step D: Finished (Truth visualized)")
    except FileNotFoundError:
        print("Step D: Files not found, skipping visual truth verification.")


    # --- 5. First prediction, Ensemble simulation ---
    ens_pred_half = np.zeros((pdim, num_ens))
    ens_pred_full = np.zeros((pdim, num_ens))
    ens_ifp_half = np.zeros((pdim, num_ens))
    ens_ifp_full = np.zeros((pdim, num_ens))

    start_time = time.time()
    workers = max(1, os.cpu_count() // 2)
    print(f"Lancement de la simulation sur {num_ens} membres ({workers} workers)...")
    
    # Saving initial ensemble and alpha_c_with_unc for MATLAB compatibility
    savemat(os.path.join(OUTPUT_DIR, 'initialEnsemble_Python.mat'), {
        'initialEnsemble': np.array(initial_ensemble, order='F'),
        'alpha_c_with_unc': np.array(alpha_c_with_unc, order='F')
    }) 
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(simulate_member_safe, alpha_c_with_unc, initial_ensemble[:, k].copy()) for k in range(num_ens)]
        for k, future in enumerate(futures):
            res_half, res_full, p_half, p_full = future.result()
            ens_pred_half[:, k] = res_half
            ens_pred_full[:, k] = res_full
            ens_ifp_half[:, k] = p_half
            ens_ifp_full[:, k] = p_full
            if (k+1) % 10 == 0:
                print(f"Terminé : {k+1}/{num_ens}")
                # Visualisation tous les 10 membres
                plot_half_full(alpha_c_with_unc, initial_ensemble[:, k], res_half, res_full, p_half, p_full, X, Y)

    print(f"Première prédiction terminée en {time.time() - start_time:.2f}s")

    # --- 5.1 Saving initial prediction results ---
    init_pred_results = {
        'ensembleOfPredictedObservations': np.array(ens_pred_half, order='F'),
        'ensembleOfIFPs': np.array(ens_ifp_half, order='F'),
        'ensembleOfPredictedObservationsFull': np.array(ens_pred_full, order='F'),
        'ensembleOfIFPsFull': np.array(ens_ifp_full, order='F'),
        'alpha_c_with_unc': alpha_c_with_unc,
        'initialEnsemble': initial_ensemble
    }
    savemat(os.path.join(OUTPUT_DIR, 'numEns100FullAddNoiseLowPl_Initial_Python.mat'), init_pred_results)
    print("File for test.py generated : numEns100FullAddNoiseLowPl_Initial_Python.mat")

    print("End of the first prediction, starting EnKF assimilation.")

    # --- 6. Enkf iteration, updating ensemble ---
    num_iterations = 4
    add_unc = 0.05

    # Loading synthetic measurements

    meas = loadmat(os.path.join(DATA_DIR, "measCancerCells.mat"))

    measurement = meas['measurement'].flatten(order='F')

    # Index selection for the top 250 variances in the ensemble predictions
    variances = np.var(ens_pred_half, axis=1)
    meas_index = np.argsort(variances)[-250:] # Top 250

    # Objective function initialization
    obj_fun = np.zeros((num_ens, num_iterations + 1))
    weight_mat = np.diag(1.0 / (add_unc**2 * np.ones(len(meas_index))))

    for k in range(num_ens):
        meas_diff = ens_pred_half[meas_index, k] - measurement[meas_index]
        obj_fun[k, 0] = meas_diff.T @ weight_mat @ meas_diff

    
    # H construction for the EnKF update
    H_indices = np.zeros((len(meas_index), 2))
    H_indices[:, 0] = np.arange(len(meas_index)) + 1 # 1-based pour enkf.py
    H_indices[:, 1] = meas_index + 1

    # W matrix for the EnKF update (observation error covariance)
    W = num_iterations * (add_unc**2) * np.eye(len(meas_index))
    
    options = {'Hones': 1, 'ignoreUninformativeMeasurements': 1}
    current_ensemble = initial_ensemble.copy()
    mean_state = None

    for i in range(num_iterations):
        print(f"Itération EnKF {i+1}/{num_iterations}")
        
        state_to_filter = np.vstack([ens_pred_half, current_ensemble])
        state_to_filter[pdim + pdim:pdim + 2*pdim, :] = np.log(state_to_filter[pdim + pdim:pdim + 2*pdim, :])
        
        # Update with the square root filter
        mean_state, updated_state = sqrt_filter(state_to_filter, W, H_indices, measurement[meas_index], options)
        
        # Extraction of the updated parameters from the updated state
        new_params = updated_state[pdim:, :]
        

        new_params[pdim:2*pdim, :] = np.exp(np.clip(new_params[pdim:2*pdim, :], 0, 5.0))

        new_params[0:pdim, :] = np.maximum(new_params[0:pdim, :], 0)

        new_params[2*pdim:3*pdim, :] *= vasc_mask[:, np.newaxis]
        new_params[4*pdim:5*pdim, :] *= growth_mask[:, np.newaxis]
        new_params[5*pdim:6*pdim, :] *= growth_mask[:, np.newaxis]

        current_ensemble = new_params
        
        # Simulation with new parameters using the faster simulator B
        print(f"Relaunching assimilation (Iter {i+1}/{num_iterations}) with simulator B...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(simulate_member_B_safe, alpha_c_with_unc, current_ensemble[:, k]) for k in range(num_ens)]
            for k, future in enumerate(futures):
                measured_values, ifp = future.result()
                ens_pred_half[:, k] = measured_values
                ens_ifp_half[:, k] = ifp

        # Objective function update after assimilation
        for k in range(num_ens):
            meas_diff = ens_pred_half[meas_index, k] - measurement[meas_index]
            obj_fun[k, i+1] = meas_diff.T @ weight_mat @ meas_diff
        
        # Intermediate results saving (updatedEnsemble and meanstate)
        intermediate_res = {
            'updatedEnsemble': np.array(current_ensemble, order='F'), 
            'meanstate': np.array(mean_state, order='F'),
        }
        savemat(os.path.join(OUTPUT_DIR, f'PlLowtempResTlConst_Python_Iter{i+1}.mat'), intermediate_res)

    # --- 7. Final saving ---
    results = {
        'alpha_c_with_unc': alpha_c_with_unc,
        'ensembleOfPredictedObservations': ens_pred_half,
        'initialEnsemble': initial_ensemble,
        'ensembleOfIFPs': ens_ifp_half,
        'ensembleOfIFPsFull': ens_ifp_full,
        'ensembleOfPredictedObservationsFull': ens_pred_full,
        'updatedEnsemble': current_ensemble,
        'objFun': obj_fun,
        'measIndex': meas_index + 1, # +1 pour compatibilité indexation MATLAB
        'meanstate': mean_state,
        'X': X, 'Y': Y
    }
    savemat(os.path.join(OUTPUT_DIR, 'PlLowfinalResAddNoise_Python_Final.mat'), results)
    print("Simulation and Assimilation completed. Results saved.")

    # --- 8. Forecast with the simulator C ---
    print("Forecast based on UPDATED mean STARTED")
    updated_mean_params = current_ensemble.mean(axis=1)
    (alpha_c_half_upd, alpha_c_full_upd, ifp_half_upd, ifp_full_upd,
     alpha_f_half_upd, alpha_f_full_upd,
     uu_w_x_half_upd, uu_w_y_half_upd,
     uu_w_x_full_upd, uu_w_y_full_upd) = three_phase_simulator_compartment_forecast(
        alpha_c_with_unc, updated_mean_params)
    print("Forecast based on UPDATED mean FINISHED")

    savemat(os.path.join(OUTPUT_DIR, 'numEns100IterationsForecastAddNoisePlLow_upd_mean_Python.mat'), {
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

    print("Forecast based on mean without updating STARTED")
    initial_mean_params = initial_ensemble.mean(axis=1)
    (alpha_c_half_mean, alpha_c_full_mean, ifp_half_mean, ifp_full_mean,
     alpha_f_half_mean, alpha_f_full_mean,
     uu_w_x_half_mean, uu_w_y_half_mean,
     uu_w_x_full_mean, uu_w_y_full_mean) = three_phase_simulator_compartment_forecast(
        alpha_c_with_unc, initial_mean_params)
    print("Forecast based on mean without updating FINISHED")

    savemat(os.path.join(OUTPUT_DIR, 'numEns100ForecastAddNoisePlLow_mean_Python.mat'), {
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
    print("Prévisions (forecast) terminées. Résultats sauvegardés.")

if __name__ == "__main__":
    t_start = time.time()
    run_ensemble_simulation()
    t_total = time.time()-t_start
    print(f"Execution time : {t_total:.2f} secondes")