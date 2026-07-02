import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_true_simulation_verify(alpha_c_half_true, ifp_half_true, alpha_c_full_true, ifp_full_true, 
                                initial_ensemble_true, X, Y, alpha_c_with_unc):
    """
    Traduction de Plot_True_simulation_Verify.m
    Visualise les données 'True' (synthétiques) pour vérification.
    """
    # Extraction explicite des dimensions de la grille
    Nx = X.shape[1]
    Ny = X.shape[0]
    dim = (Ny, Nx)
    pdim = Nx * Ny
    eps = 0.01

    # Extraction et redimensionnement des paramètres 'True'
    # Ordre : alpF, khat_w, T_vess, T_lymp, Kgrow, Thetagrow
    alpC0 = alpha_c_with_unc.reshape(dim)
    alpF0 = initial_ensemble_true[0:pdim].reshape(dim)
    khat_w = initial_ensemble_true[pdim:2*pdim].reshape(dim)
    T_vess = initial_ensemble_true[2*pdim:3*pdim].reshape(dim)
    T_lymp = initial_ensemble_true[3*pdim:4*pdim].reshape(dim)
    Kgrow = initial_ensemble_true[4*pdim:5*pdim].reshape(dim)
    Thetagrow = initial_ensemble_true[5*pdim:6*pdim].reshape(dim)

    # Redimensionnement des résultats de simulation 'True'
    alpC_half = alpha_c_half_true.reshape(dim)
    ifp_half = ifp_half_true.reshape(dim)
    alpC_full = alpha_c_full_true.reshape(dim)
    ifp_full = ifp_full_true.reshape(dim)

    fig = plt.figure(figsize=(16, 12))
    plt.ion()

    def draw_plots(alpC_curr, ifp_curr, label):
        fig.clf()
        
        # 1. Initial Cell Volume Fraction
        ax1 = fig.add_subplot(3, 3, 1)
        im1 = ax1.pcolormesh(X, Y, alpC0, cmap='viridis', vmin=0, vmax=0.3, shading='gouraud')
        ax1.contour(X, Y, alpC0, [eps], colors='white', linewidths=0.5)
        ax1.set_title("True alpC Initial")
        fig.colorbar(im1, ax=ax1)

        # 2. Initial Fibroblast Volume Fraction
        ax2 = fig.add_subplot(3, 3, 2)
        im2 = ax2.pcolormesh(X, Y, alpF0, cmap='viridis', vmin=0, vmax=0.3, shading='gouraud')
        ax2.set_title("True alpF Initial")
        fig.colorbar(im2, ax=ax2)

        # 3. log(khat_w)
        ax3 = fig.add_subplot(3, 3, 3)
        im3 = ax3.pcolormesh(X, Y, np.log(khat_w), cmap='viridis', vmin=1, vmax=5, shading='gouraud')
        ax3.set_title("True log(khat_w)")
        fig.colorbar(im3, ax=ax3)

        # 4. Cell Volume Fraction (Current stage)
        ax4 = fig.add_subplot(3, 3, 4)
        im4 = ax4.pcolormesh(X, Y, alpC_curr, cmap='viridis', vmin=0, vmax=0.3, shading='gouraud')
        ax4.contour(X, Y, alpC_curr, [eps], colors='white', linewidths=0.5)
        ax4.set_title(f"True alpC ({label})")
        fig.colorbar(im4, ax=ax4)

        # 5. Kgrow
        ax5 = fig.add_subplot(3, 3, 5)
        im5 = ax5.pcolormesh(X, Y, Kgrow, cmap='viridis', vmin=0, vmax=10, shading='gouraud')
        ax5.set_title("True Kgrow")
        fig.colorbar(im5, ax=ax5)

        # 6. T_lymp
        ax6 = fig.add_subplot(3, 3, 6)
        im6 = ax6.pcolormesh(X, Y, T_lymp, cmap='viridis', vmin=0, vmax=0.006, shading='gouraud')
        ax6.set_title("True T_lymp")
        fig.colorbar(im6, ax=ax6)

        # 7. IFP (Pression interstitielle - 3D)
        ax7 = fig.add_subplot(3, 3, 7, projection='3d')
        ifp_converted = (ifp_curr - 101325) / 133
        surf = ax7.plot_surface(X, Y, ifp_converted, cmap='viridis', edgecolor='none')
        ax7.set_zlim(-5, max(50, np.max(ifp_converted)))
        ax7.set_title(f"True IFP ({label})")
        ax7.view_init(elev=30, azim=60)
        fig.colorbar(surf, ax=ax7, shrink=0.5, aspect=5)

        # 8. Thetagrow
        ax8 = fig.add_subplot(3, 3, 8)
        im8 = ax8.pcolormesh(X, Y, Thetagrow, cmap='viridis', vmin=0, vmax=10, shading='gouraud')
        ax8.set_title("True Thetagrow")
        fig.colorbar(im8, ax=ax8)

        # 9. T_vess
        ax9 = fig.add_subplot(3, 3, 9)
        im9 = ax9.pcolormesh(X, Y, T_vess, cmap='viridis', vmin=0, vmax=0.005, shading='gouraud')
        ax9.set_title("True T_vess")
        fig.colorbar(im9, ax=ax9)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)

    # Affichage des données à mi-parcours (Half)
    draw_plots(alpC_half, ifp_half, "Half")
    plt.pause(2)

    # Mise à jour avec les données finales (Full)
    draw_plots(alpC_full, ifp_full, "Full")
    plt.pause(2)