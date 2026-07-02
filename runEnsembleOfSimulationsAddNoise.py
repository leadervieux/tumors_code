import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import rotate, gaussian_filter
from scipy.io import savemat, loadmat
import time
from concurrent.futures import ProcessPoolExecutor

# Import des modules locaux
from fastGaussian import fast_gaussian
# On importe la fonction du simulateur depuis le fichier que tu as gardé
from A_three_phase_simulator_compartment_full_May08_growth_opt import three_phase_simulator_compartment_full
# On s'assure que le nom du fichier du filtre est correct (enkf.py ou sqrtFilter.py)
from sqrtFilter import sqrt_filter
from Plot_EnsemblePred_Iteration_0_HalfFull import plot_half_full
from Plot_True_simulation_Verify import plot_true_simulation_verify

def simulate_member_safe(alpha_c, member_params, member_idx=0):
    """
    Wrapper pour gérer l'adaptativité de NTime (Inc_dt) comme en MATLAB.
    """
    nan_result = True
    inc_dt = 0
    while nan_result and inc_dt < 5: # Limite à 5 essais pour éviter les boucles infinies
        res_half, res_full, p_half, p_full = three_phase_simulator_compartment_full(alpha_c, member_params, inc_dt, member_idx)
        
        if np.isnan(p_full).any():
            inc_dt += 1
            print(f"Instabilité détectée. Relance avec Inc_dt={inc_dt}")
        else:
            nan_result = False
    return res_half, res_full, p_half, p_full

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

    # Définition des positions vasculaires et lymphatiques
    vascPos = np.ones(dim)
    lymphPos = np.ones(dim)
    blankLymphPos = (X - 0.5)**2 + (Y - 0.5)**2 < 0.25**2
    lymphPos = lymphPos - blankLymphPos.astype(float)

    # --- 3. Configuration de l'Ensemble ---
    #num_ens = 100
    # Le vecteur d'état contient 6 champs de taille pdim
    data = loadmat('new_initial_ensemble_Python_1.mat')

    initial_ensemble = data['initialEnsemble']
    alpha_c_with_unc = data['alpha_c_with_unc']

    num_ens = initial_ensemble.shape[1]

    print(f"Forme de initial_ensemble : {initial_ensemble.shape}")
    print(f"Forme de alpha_c_with_unc : {alpha_c_with_unc.shape}")

    print("Étape C : Ensemble initial généré")

    # --- 4. Step D: run true model / Loading Truth ---
    initial_true_model = None
    true_model_data_half = None
    try:
        true_init = loadmat('Ensemble_Initial_E20_True_April14_theta0_K0.mat')
        true_sol = loadmat('Ensemble_Solution_E20_True_Growth_May08_vary_theta0_K0_reduced.mat')
        
        initial_ensemble_true = true_init['initialEnsemble_True']
        # On choisit le membre K=7 comme vérité (index 6 en Python)
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


    # --- 5. Simulation de l'ensemble (Première prédiction) ---
    ens_pred_half = np.zeros((pdim, num_ens))
    ens_pred_full = np.zeros((pdim, num_ens))
    ens_ifp_half = np.zeros((pdim, num_ens))
    ens_ifp_full = np.zeros((pdim, num_ens))

    start_time = time.time()
    # On limite le nombre de travailleurs à la moitié des coeurs physiques pour éviter la saturation RAM
    workers = max(1, os.cpu_count() // 2)
    print(f"Lancement de la simulation sur {num_ens} membres ({workers} workers)...")
    
    # Sauvegarde groupée pour MATLAB
    savemat('initialEnsemble_Python.mat', {
        'initialEnsemble': np.array(initial_ensemble, order='F'),
        'alpha_c_with_unc': np.array(alpha_c_with_unc, order='F')
    }) 
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(simulate_member_safe, alpha_c_with_unc, initial_ensemble[:, k].copy(), k) for k in range(num_ens)]
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

    # --- 5.1 Sauvegarde de la première prédiction (Comme dans MATLAB) ---
    # Utiliser order='F' pour que MATLAB lise les vecteurs correctement
    init_pred_results = {
        'ensembleOfPredictedObservations': np.array(ens_pred_half, order='F'),
        'ensembleOfIFPs': np.array(ens_ifp_half, order='F'),
        'ensembleOfPredictedObservationsFull': np.array(ens_pred_full, order='F'),
        'ensembleOfIFPsFull': np.array(ens_ifp_full, order='F'),
        'alpha_c_with_unc': alpha_c_with_unc,
        'initialEnsemble': initial_ensemble
    }
    savemat('numEns100FullAddNoiseLowPl_Initial_Python.mat', init_pred_results)
    print("Fichier pour test.py généré : numEns100FullAddNoiseLowPl_Initial_Python.mat")

    # --- MODE TEST : On s'arrête ici pour comparer avec MATLAB via test.py ---
    print("Mode test activé : Fin du script avant l'assimilation EnKF.")
    return 

    # --- 6. Itérations EnKF (Mise à jour des paramètres) ---
    num_iterations = 4
    add_unc = 0.05

    # Création d'une mesure synthétique
    # On prend la prédiction 'Half' du membre vérité + un bruit gaussien
    if true_model_data_half is not None:
        true_res_half = true_model_data_half[0:pdim]
    else:
        true_res_half = ens_pred_half[:, 0] # Fallback si le Step D a échoué

    measurement = true_res_half + add_unc * np.random.randn(pdim)
    measurement[measurement < 0] = 0

    # Sélection des indices à haute variance pour l'assimilation (comme dans MATLAB)
    variances = np.var(ens_pred_half, axis=1)
    meas_index = np.argsort(variances)[-250:] # Top 250

    # Initialisation de la fonction de coût (objFun)
    obj_fun = np.zeros((num_ens, num_iterations + 1))
    weight_mat = np.diag(1.0 / (add_unc**2 * np.ones(len(meas_index))))

    for k in range(num_ens):
        meas_diff = ens_pred_half[meas_index, k] - measurement[meas_index]
        obj_fun[k, 0] = meas_diff.T @ weight_mat @ meas_diff

    
    # Construction de H (opérateur d'observation)
    H_indices = np.zeros((len(meas_index), 2))
    H_indices[:, 0] = np.arange(len(meas_index)) + 1 # 1-based pour enkf.py
    H_indices[:, 1] = meas_index + 1

    # Matrice de bruit de mesure W
    W = num_iterations * (add_unc**2) * np.eye(len(meas_index))
    
    options = {'Hones': 1, 'ignoreUninformativeMeasurements': 1}
    current_ensemble = initial_ensemble.copy()
    mean_state = None

    for i in range(num_iterations):
        print(f"Itération EnKF {i+1}/{num_iterations}")
        
        # Préparation du vecteur d'état pour le filtre [F(X); X]
        # On passe khat_w (index pdim:2*pdim) en log pour le filtre
        state_to_filter = np.vstack([ens_pred_half, current_ensemble])
        state_to_filter[pdim + pdim:pdim + 2*pdim, :] = np.log(state_to_filter[pdim + pdim:pdim + 2*pdim, :])
        
        # Mise à jour par le filtre Square Root
        mean_state, updated_state = sqrt_filter(state_to_filter, W, H_indices, measurement[meas_index], options)
        
        # Extraction et corrections physiques des paramètres mis à jour
        new_params = updated_state[pdim:, :]
        
        # Correction log(khat_w) -> khat_w
        new_params[pdim:2*pdim, :] = np.exp(np.clip(new_params[pdim:2*pdim, :], 0, 5.0))
        # Correction alpF
        new_params[0:pdim, :] = np.maximum(new_params[0:pdim, :], 0)
        # Application des masques vasculaires sur Tvess et Kgrow
        new_params[2*pdim:3*pdim, :] *= vasc_mask[:, np.newaxis]
        new_params[4*pdim:5*pdim, :] *= growth_mask[:, np.newaxis]
        new_params[5*pdim:6*pdim, :] *= growth_mask[:, np.newaxis]

        current_ensemble = new_params
        
        # Relancer la simulation avec les nouveaux paramètres pour la prochaine itération
        print(f"Relance de l'assimilation (Iter {i+1}/4)...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(simulate_member_safe, alpha_c_with_unc, current_ensemble[:, k]) for k in range(num_ens)]
            for k, future in enumerate(futures):
                res_half, res_full, p_half, p_full = future.result()
                ens_pred_half[:, k] = res_half
                ens_pred_full[:, k] = res_full
                ens_ifp_half[:, k] = p_half
                ens_ifp_full[:, k] = p_full

        # Calcul de la fonction de coût après mise à jour
        for k in range(num_ens):
            meas_diff = ens_pred_half[meas_index, k] - measurement[meas_index]
            obj_fun[k, i+1] = meas_diff.T @ weight_mat @ meas_diff
        
        # Sauvegarde intermédiaire (Optionnel, comme en MATLAB)
        intermediate_res = {
            'updatedEnsemble': np.array(current_ensemble, order='F'), 
            'meanstate': np.array(mean_state, order='F'),
            'ensembleOfPredictedObservations': np.array(ens_pred_half, order='F'),
            'ensembleOfPredictedObservationsFull': np.array(ens_pred_full, order='F'),
            'ensembleOfIFPs': np.array(ens_ifp_half, order='F'),
            'ensembleOfIFPsFull': np.array(ens_ifp_full, order='F')
        }
        savemat(f'PlLowtempResTlConst_Python_Iter{i+1}.mat', intermediate_res)

    # --- 7. Sauvegarde finale ---
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
    savemat('PlLowfinalResAddNoise_Python_Final.mat', results)
    print("Simulation et Assimilation terminées. Résultats sauvegardés.")

if __name__ == "__main__":
    run_ensemble_simulation()