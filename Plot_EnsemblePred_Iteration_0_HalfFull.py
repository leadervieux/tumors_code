import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_half_full(alpha_c_with_unc, initial_ensemble, measured_values_half, measured_values_full, ifp_half, ifp_full, X, Y):
    """
    Traduction de Plot_EnsemblePred_Iteration_0_HalfFull.m
    Affiche l'état initial et les prédictions (Half/Full) de la tumeur.
    """
    Nx = X.shape[1]
    Ny = X.shape[0]
    dim = (Ny, Nx)
    pdim = np.prod(dim)
    eps = 0.01

    # Extraction et redimensionnement des paramètres de l'ensemble
    # On suit l'ordre du vecteur d'état MATLAB (1:prod(dim), prod(dim)+1:2*prod(dim), etc.)
    alpC0 = alpha_c_with_unc.reshape(dim)
    alpF0 = initial_ensemble[0:pdim].reshape(dim)
    khat_w = initial_ensemble[pdim:2*pdim].reshape(dim)
    T_vess = initial_ensemble[2*pdim:3*pdim].reshape(dim)
    T_lymp = initial_ensemble[3*pdim:4*pdim].reshape(dim)
    Kgrow = initial_ensemble[4*pdim:5*pdim].reshape(dim)
    Thetagrow = initial_ensemble[5*pdim:6*pdim].reshape(dim)

    # Redimensionnement des résultats de simulation
    alpC_half = measured_values_half.reshape(dim)
    alpC_full = measured_values_full.reshape(dim)
    ifp_half_val = ifp_half.reshape(dim)
    ifp_full_val = ifp_full.reshape(dim)

    # Configuration de la figure
    fig = plt.figure(figsize=(16, 12))
    plt.ion() # Mode interactif pour permettre l'affichage pendant la boucle

    def draw_all(alpC_pred, ifp_pred, label):
        fig.clf()
        
        # 1. Cell Volume Fraction (Initial)
        ax1 = fig.add_subplot(3, 3, 1)
        im1 = ax1.pcolormesh(X, Y, alpC0, cmap='viridis', vmin=0, vmax=0.3, shading='gouraud')
        ax1.contour(X, Y, alpC0, [eps], colors='white', linewidths=0.5)
        ax1.set_title("Initial Cell Volume Fraction")
        fig.colorbar(im1, ax=ax1)

        # 2. Fibroblast Volume Fraction (Initial)
        ax2 = fig.add_subplot(3, 3, 2)
        im2 = ax2.pcolormesh(X, Y, alpF0, cmap='viridis', vmin=0, vmax=0.3, shading='gouraud')
        ax2.set_title("Fibroblast Volume Fraction")
        fig.colorbar(im2, ax=ax2)

        # 3. log(khat_w)
        ax3 = fig.add_subplot(3, 3, 3)
        im3 = ax3.pcolormesh(X, Y, np.log(khat_w), cmap='viridis', vmin=1, vmax=4, shading='gouraud')
        ax3.set_title("log(khat_w)")
        fig.colorbar(im3, ax=ax3)

        # 4. Cell Volume Fraction (Prediction)
        ax4 = fig.add_subplot(3, 3, 4)
        im4 = ax4.pcolormesh(X, Y, alpC_pred, cmap='viridis', vmin=0, vmax=0.3, shading='gouraud')
        ax4.contour(X, Y, alpC_pred, [eps], colors='white', linewidths=0.5)
        ax4.set_title(f"Cell Volume Fraction ({label})")
        fig.colorbar(im4, ax=ax4)

        # 5. Growth Rate (Kgrow)
        ax5 = fig.add_subplot(3, 3, 5)
        im5 = ax5.pcolormesh(X, Y, Kgrow, cmap='viridis', vmin=0, vmax=10, shading='gouraud')
        ax5.set_title("Kgrow (Growth/Death)")
        fig.colorbar(im5, ax=ax5)

        # 6. T_lymp (Lymphatic filtration)
        ax6 = fig.add_subplot(3, 3, 6)
        im6 = ax6.pcolormesh(X, Y, T_lymp, cmap='viridis', vmin=0, vmax=0.006, shading='gouraud')
        ax6.set_title("T_lymp")
        fig.colorbar(im6, ax=ax6)

        # 7. IFP (Pression interstitielle - 3D)
        ax7 = fig.add_subplot(3, 3, 7, projection='3d')
        ifp_converted = (ifp_pred - 101325) / 133
        surf = ax7.plot_surface(X, Y, ifp_converted, cmap='viridis', edgecolor='none')
        ax7.set_zlim(-5, 50)
        ax7.set_title(f"IFP ({label})")
        ax7.view_init(elev=30, azim=60)
        ax7.set_xlabel('X')
        ax7.set_ylabel('Y')

        # 8. Capacity (Thetagrow)
        ax8 = fig.add_subplot(3, 3, 8)
        im8 = ax8.pcolormesh(X, Y, Thetagrow, cmap='viridis', vmin=0, vmax=10, shading='gouraud')
        ax8.set_title("Thetagrow")
        fig.colorbar(im8, ax=ax8)

        # 9. T_vess (Vascular filtration)
        ax9 = fig.add_subplot(3, 3, 9)
        im9 = ax9.pcolormesh(X, Y, T_vess, cmap='viridis', vmin=0, vmax=0.005, shading='gouraud')
        ax9.set_title("T_vess")
        fig.colorbar(im9, ax=ax9)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)

    # Affichage de la prédiction à mi-parcours (Half)
    draw_all(alpC_half, ifp_half_val, "Half")
    print("Affichage Prediction Half...")
    plt.pause(0.1)

    # Mise à jour avec la prédiction finale (Full)
    draw_all(alpC_full, ifp_full_val, "Full")
    print("Affichage Prediction Full.")
    plt.pause(1)