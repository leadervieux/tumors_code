import numpy as np
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded

def solve_pressure_sparse(pL_guess, lam_T_half_Lef, lam_T_half_Rig, lam_T_half_Bot, lam_T_half_Top,
                          Pstar_Lef, Pstar_Rig, Pstar_Bot, Pstar_Top, PvStar, PlStar, Tv, Tl,
                          i_dx2, i_dy2, Nx, Ny):

    p_ref = np.ones((Ny, Nx)) * 101325.0

    P_1   = np.zeros((Ny, Nx))
    P_old = pL_guess.copy()
    P     = np.zeros((Ny, Nx))

    T_param = 1000000 * 10
    NT      = 200 * 1000
    dt      = T_param / NT

    lamb_x = dt * i_dx2
    lamb_y = dt * i_dy2

    Q_source_b = Tv * PvStar + Tl * PlStar
    Q_source_A = Tv + Tl

    Error = 1.0

    while Error > 0.000005:

        # --- Demi-pas X : boucle sur chaque ligne iy ---
        for iy in range(Ny):

            # Construction de la matrice tridiagonale SS_x (taille Nx x Nx)
            main  = np.zeros(Nx)
            upper = np.zeros(Nx - 1)  # sur-diagonale
            lower = np.zeros(Nx - 1)  # sous-diagonale
            F_help_diag = np.zeros(Nx)  # diagonale de SS_x_help

            # Cellule bord gauche (ix=0)
            main[0]       = 1 + 0.5*lamb_x*(2*lam_T_half_Lef[iy,0] + lam_T_half_Rig[iy,0]) + dt*Q_source_A[iy,0]
            upper[0]      = -0.5*lamb_x*lam_T_half_Rig[iy,0]
            F_help_diag[0] = main[0] - dt*Q_source_A[iy,0]

            # Cellules intérieures
            for ii in range(1, Nx-1):
                lower[ii-1]     = -0.5*lamb_x*lam_T_half_Lef[iy,ii]
                main[ii]        = 1 + 0.5*lamb_x*(lam_T_half_Lef[iy,ii] + lam_T_half_Rig[iy,ii]) + dt*Q_source_A[iy,ii]
                upper[ii]       = -0.5*lamb_x*lam_T_half_Rig[iy,ii]
                F_help_diag[ii] = main[ii] - dt*Q_source_A[iy,ii]

            # Cellule bord droit (ix=Nx-1)
            lower[Nx-2]         = -0.5*lamb_x*lam_T_half_Lef[iy,Nx-1]
            main[Nx-1]          = 1 + 0.5*lamb_x*(lam_T_half_Lef[iy,Nx-1] + 2*lam_T_half_Rig[iy,Nx-1]) + dt*Q_source_A[iy,Nx-1]
            F_help_diag[Nx-1]   = main[Nx-1] - dt*Q_source_A[iy,Nx-1]

            # SS_x_help = I - SS_x_help_sparse  (comme MATLAB diag - sparse)
            # En pratique : SS_x_help * P_old[iy] = (I - F_help) * P_old[iy]
            # = P_old[iy] - F_help_diag*P_old[iy] - upper_offdiag*P_old - lower_offdiag*P_old
            # On calcule SS_x_help * P_old[iy,:] directement
            P_row = P_old[iy, :]
            SS_x_help_times_P = P_row.copy()
            SS_x_help_times_P       -= F_help_diag * P_row
            SS_x_help_times_P[:-1]  -= upper * P_row[1:]
            SS_x_help_times_P[1:]   -= lower * P_row[:-1]

            # Construction du RHS bb_x
            bb_x = np.zeros(Nx)
            if iy == 0:
                bb_x = (P_old[iy,:]  * (1 - lamb_y*(2*lam_T_half_Bot[0,:] + lam_T_half_Top[0,:]))
                      + lamb_y * lam_T_half_Top[0,:] * P_old[iy+1,:]
                      + lamb_y * 2*lam_T_half_Bot[0,:] * Pstar_Bot[0,:]
                      + dt * Q_source_b[iy,:])
            elif iy == Ny-1:
                bb_x = (P_old[iy,:] * (1 - lamb_y*(lam_T_half_Bot[Ny-1,:] + 2*lam_T_half_Top[Ny-1,:]))
                      + lamb_y * lam_T_half_Bot[Ny-1,:] * P_old[iy-1,:]
                      + lamb_y * 2*lam_T_half_Top[Ny-1,:] * Pstar_Top[0,:]
                      + dt * Q_source_b[iy,:])
            else:
                bb_x = (P_old[iy,:] * (1 - lamb_y*(lam_T_half_Bot[iy,:] + lam_T_half_Top[iy,:]))
                      + lamb_y * lam_T_half_Bot[iy,:] * P_old[iy-1,:]
                      + lamb_y * lam_T_half_Top[iy,:] * P_old[iy+1,:]
                      + dt * Q_source_b[iy,:])

            # Correction D-G
            bb_x += SS_x_help_times_P

            # Correction bords gauche/droit
            bb_x[0]    += lamb_x * 2 * lam_T_half_Lef[iy,0]    * Pstar_Lef[iy,0]
            bb_x[Nx-1] += lamb_x * 2 * lam_T_half_Rig[iy,Nx-1] * Pstar_Rig[iy,0]

            # Résolution tridiagonale (format banded SciPy)
            ab = np.zeros((3, Nx))
            ab[0, 1:]  = upper   # sur-diagonale
            ab[1, :]   = main    # diagonale
            ab[2, :-1] = lower   # sous-diagonale
            
            # Protection contre les pivots nuls
            ab[1, ab[1, :] == 0] = 1e-12
            
            P_1[iy, :] = solve_banded((1, 1), ab, bb_x)

        # --- Demi-pas Y : boucle sur chaque colonne ix ---
        for ix in range(Nx):

            main  = np.zeros(Ny)
            upper = np.zeros(Ny - 1)
            lower = np.zeros(Ny - 1)

            # Cellule bord bas (iy=0)
            main[0]  = 1 + 0.5*lamb_y*(2*lam_T_half_Bot[0,ix] + lam_T_half_Top[0,ix])
            upper[0] = -0.5*lamb_y*lam_T_half_Top[0,ix]

            # Cellules intérieures
            for ii in range(1, Ny-1):
                lower[ii-1] = -0.5*lamb_y*lam_T_half_Bot[ii,ix]
                main[ii]    = 1 + 0.5*lamb_y*(lam_T_half_Bot[ii,ix] + lam_T_half_Top[ii,ix])
                upper[ii]   = -0.5*lamb_y*lam_T_half_Top[ii,ix]

            # Cellule bord haut (iy=Ny-1)
            lower[Ny-2] = -0.5*lamb_y*lam_T_half_Bot[Ny-1,ix]
            main[Ny-1]  = 1 + 0.5*lamb_y*(lam_T_half_Bot[Ny-1,ix] + 2*lam_T_half_Top[Ny-1,ix])

            # SS_y_help = SS_y - I  → SS_y_help * P_old[:,ix]
            P_col = P_old[:, ix]
            SS_y_help_times_P = main * P_col - P_col
            SS_y_help_times_P[:-1] += upper * P_col[1:]
            SS_y_help_times_P[1:]  += lower * P_col[:-1]

            bb_y = P_1[:, ix] + SS_y_help_times_P

            # Résolution tridiagonale
            ab = np.zeros((3, Ny))
            ab[0, 1:]  = upper
            ab[1, :]   = main
            ab[2, :-1] = lower
            
            ab[1, ab[1, :] == 0] = 1e-12
            
            P[:, ix] = solve_banded((1, 1), ab, bb_y)

        # Critère de convergence (même formule que MATLAB norm_2D)
        a = np.sum(np.abs(P - P_old))
        b = np.sum(p_ref)
        Error = a / b
        P_old = P.copy()

    return P