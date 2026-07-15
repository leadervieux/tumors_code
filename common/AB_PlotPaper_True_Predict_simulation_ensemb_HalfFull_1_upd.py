import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import label, regionprops

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "common"))
sys.path.insert(0, THIS_DIR)

from paths import DATA_DIR, output_dir
OUTPUT_DIR = output_dir(THIS_DIR)


def plot_paper_comparison(K_member):
    """
    Translated from AB_PlotPaper_True_Predict_simulation_ensemb_HalfFull_1_upd.m
    Visualizes the comparison between the synthetic data and the predictions.
    """
    # --- 1. Loading 'True' data (Synthetic) ---
    try:
        true_init = loadmat(os.path.join(DATA_DIR, 'Ensemble_Initial_E20_True_April14_theta0_K0.mat'))
        true_sol = loadmat(os.path.join(DATA_DIR, 'Ensemble_Solution_E20_True_Growth_May08_vary_theta0_K0_reduced.mat'))
        
        X = true_init['X']
        Y = true_init['Y']
        alpha_c_with_unc = true_init['alpha_c_with_unc']
        initialEnsemble_True = true_init['initialEnsemble_True']
        
        # Temporal solutions (Half / Full)
        alpha_c_Half_True = true_sol['alpha_c_Half_True']
        alpha_f_Half_True = true_sol['alpha_f_Half_True']
        IFP_Half_True = true_sol['IFP_Half_True']
        uu_W_x_Half_True = true_sol['uu_W_x_Half_True']
        uu_W_y_Half_True = true_sol['uu_W_y_Half_True']
        
        alpha_c_Full_True = true_sol['alpha_c_Full_True']
        alpha_f_Full_True = true_sol['alpha_f_Full_True']
        IFP_Full_True = true_sol['IFP_Full_True']
        uu_W_x_Full_True = true_sol['uu_W_x_Full_True']
        uu_W_y_Full_True = true_sol['uu_W_y_Full_True']
    except FileNotFoundError as e:
        print(f"Erreur : Fichier de données vérité manquant. {e}")
        return

    Nx, Ny = X.shape[1], X.shape[0]
    dim = (Ny, Nx)
    eps = 0.01

    # --- 2. Loading 'Estimated' data (EnKF) ---
    try:
        enkf_res = loadmat(os.path.join(OUTPUT_DIR, 'PlLowfinalResAddNoise_May08-Mar17_K0_theta0_addunc_0p05_C7_249.mat'))
        enkf_pred = loadmat(os.path.join(OUTPUT_DIR, 'numEns100IterationsForecastAddNoisePlLow_upd_mean_May08-Mar17_K0_theta0_addunc_0p05_C7_249.mat'))
        
        updatedEnsemble = enkf_res['updatedEnsemble']
        
        # Means of the updated predictions
        alpC_upd = enkf_pred['alpha_c_Half_upd_mean'].reshape(dim)
        alpF_upd = enkf_pred['alpha_f_Half_upd_mean'].reshape(dim)
        ifp_upd = enkf_pred['IFP_Half_upd_mean'].reshape(dim)
        uWx_upd = enkf_pred['uu_W_x_Half_upd_mean'].reshape(dim)
        uWy_upd = enkf_pred['uu_W_y_Half_upd_mean'].reshape(dim)

        # Full data for the second phase of the plot
        alpC_full_upd = enkf_pred['alpha_c_Full_upd_mean'].reshape(dim)
        ifp_full_upd = enkf_pred['IFP_Full_upd_mean'].reshape(dim)
    except FileNotFoundError as e:
        print(f"Erreur : Fichier de résultats EnKF manquant. {e}")
        return

    fig = plt.figure(figsize=(18, 14))
    plt.ion()

    def draw_comparison(stage_label, alpC_T, alpF_T, ifp_T, uWx_T, uWy_T, alpC_E, alpF_E, ifp_E, uWx_E, uWy_E):
        fig.clf()
        
        # --- COLUMN 1 : SYNTHETIC DATA ---
        # Cancer Cells
        ax1 = fig.add_subplot(4, 3, 1)
        im1 = ax1.pcolormesh(X, Y, alpC_T, cmap='viridis', vmin=0, vmax=0.3, shading='gouraud')
        ax1.contour(X, Y, alpC_T, [eps], colors='white')
        ax1.set_title("Synthetic: Cancer Cell")
        fig.colorbar(im1, ax=ax1)

        # Fibroblasts
        ax4 = fig.add_subplot(4, 3, 4)
        im4 = ax4.pcolormesh(X, Y, alpF_T, cmap='viridis', vmin=0, vmax=0.3, shading='gouraud')
        ax4.set_title("Synthetic: Fibroblast")
        fig.colorbar(im4, ax=ax4)

        # IFP (3D)
        ax7 = fig.add_subplot(4, 3, 7, projection='3d')
        ifp_t_conv = (ifp_T - 101325) / 133
        ax7.plot_surface(X, Y, ifp_t_conv, cmap='viridis', edgecolor='none')
        ax7.set_title("Synthetic: IFP (mmHg)")
        ax7.view_init(30, 30)

        # Velocity (Quiver + Heatmap)
        ax10 = fig.add_subplot(4, 3, 10)
        speed_t = np.sqrt(uWx_T**2 + uWy_T**2)
        ax10.pcolormesh(X, Y, speed_t, cmap='magma', vmin=0, vmax=0.4, alpha=0.3)
        ax10.quiver(X[::3, ::3], Y[::3, ::3], uWx_T[::3, ::3], uWy_T[::3, ::3], color='black')
        ax10.set_title("Synthetic: IF Velocity")

        # --- COLUMN 2 : PREDICTED BEHAVIOR ---
        # Cancer Cells
        ax2 = fig.add_subplot(4, 3, 2)
        im2 = ax2.pcolormesh(X, Y, alpC_E, cmap='viridis', vmin=0, vmax=0.3, shading='gouraud')
        ax2.contour(X, Y, alpC_E, [eps], colors='white')
        ax2.set_title("Predicted Behavior")
        fig.colorbar(im2, ax=ax2)

        # Fibroblasts
        ax5 = fig.add_subplot(4, 3, 5)
        im5 = ax5.pcolormesh(X, Y, alpF_E, cmap='viridis', vmin=0, vmax=0.3, shading='gouraud')
        ax5.set_title("Predicted Fibroblast")
        fig.colorbar(im5, ax=ax5)

        # IFP Predicted
        ax8 = fig.add_subplot(4, 3, 8, projection='3d')
        ifp_e_conv = (ifp_E - 101325) / 133
        ax8.plot_surface(X, Y, ifp_e_conv, cmap='viridis', edgecolor='none')
        ax8.view_init(30, 30)

        # Velocity Predicted
        ax11 = fig.add_subplot(4, 3, 11)
        speed_e = np.sqrt(uWx_E**2 + uWy_E**2)
        ax11.pcolormesh(X, Y, speed_e, cmap='magma', vmin=0, vmax=0.4, alpha=0.3)
        ax11.quiver(X[::3, ::3], Y[::3, ::3], uWx_E[::3, ::3], uWy_E[::3, ::3])

        # --- COLUMN 3 : ERROR (Diff) ---
        # Error Cells
        ax3 = fig.add_subplot(4, 3, 3)
        im3 = ax3.pcolormesh(X, Y, alpC_E - alpC_T, cmap='seismic', vmin=-0.1, vmax=0.1, shading='gouraud')
        ax3.set_title("Error: Cancer Cells")
        fig.colorbar(im3, ax=ax3)

        # Error Fibroblasts
        ax6 = fig.add_subplot(4, 3, 6)
        im6 = ax6.pcolormesh(X, Y, alpF_E - alpF_T, cmap='seismic', vmin=-0.1, vmax=0.1, shading='gouraud')
        fig.colorbar(im6, ax=ax6)

        # Error IFP
        ax9 = fig.add_subplot(4, 3, 9)
        im9 = ax9.pcolormesh(X, Y, (ifp_E - ifp_T)/133, cmap='seismic', vmin=-10, vmax=10, shading='gouraud')
        fig.colorbar(im9, ax=ax9)
        
        # Error Velocity (Magnitude diff)
        ax12 = fig.add_subplot(4, 3, 12)
        err_vel = np.sqrt((uWx_E - uWx_T)**2 + (uWy_E - uWy_T)**2)
        im12 = ax12.pcolormesh(X, Y, err_vel, cmap='Reds', vmin=0, vmax=0.2, shading='gouraud')
        fig.colorbar(im12, ax=ax12)

        # Stats tumeur (skimage regionprops)
        binary_tumor = alpC_E >= eps
        label_img = label(binary_tumor)
        props = regionprops(label_img)
        if props:
            largest = max(props, key=lambda p: p.area)
            print(f"[{stage_label}] Area: {largest.area}, Perimeter: {largest.perimeter:.2f}")

        plt.suptitle(f"Ensemble Update Comparison - Member {K_member} - {stage_label}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.draw()
        plt.pause(0.1)

    # --- Plotting step 1 : Half Time ---
    K = K_member - 1 # Index Python
    draw_comparison("HALF TIME", 
                    alpha_c_Half_True[:, K].reshape(dim), alpha_f_Half_True[:, K].reshape(dim), 
                    IFP_Half_True[:, K].reshape(dim), uu_W_x_Half_True[:, K].reshape(dim), uu_W_y_Half_True[:, K].reshape(dim),
                    alpC_upd, alpF_upd, ifp_upd, uWx_upd, uWy_upd)
    
    print("Affichage Half Time terminé. Pause de 5 secondes.")
    plt.pause(5)

    # --- Plotting step 2 : Full Time ---
    draw_comparison("FULL TIME", 
                    alpha_c_Full_True[:, K].reshape(dim), alpha_f_Full_True[:, K].reshape(dim), 
                    IFP_Full_True[:, K].reshape(dim), uu_W_x_Full_True[:, K].reshape(dim), uu_W_y_Full_True[:, K].reshape(dim),
                    alpC_full_upd, alpF_upd, ifp_full_upd, uWx_upd, uWy_upd)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    # Test the plotting function with a specific ensemble member
    plot_paper_comparison(7)