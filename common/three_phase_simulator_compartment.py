import numpy as np
import os
from scipy.io import savemat
from scipy.sparse import diags, csc_matrix, eye as speye
from scipy.sparse.linalg import spsolve, factorized
from solve_Pressure_sparse import solve_pressure_sparse
from lambda_chem import lambda_chem
from lambda_TGF import lambda_TGF
from source_cell_theta import source_cell_theta
from source_fibroblast_new import source_fibroblast_new
from paths import REPO_ROOT


_DEBUG_DIR = os.path.join(REPO_ROOT, "debug")
os.makedirs(_DEBUG_DIR, exist_ok=True)

def _three_phase_simulator_compartment_core(alpha_c, initialEnsemble, NTime, T_phys,
                                              capture_half=True, extended_outputs=False):
    #####################
    # 1. Discretization #
    #####################

    ###########################
    # 1.1 Time discretization #
    ###########################

    # Computations
    NTime = int(NTime)

    ###########################
    # 1.2 Grid discretization #
    ###########################
    Nx = 61  # number of grid cells x dir
    Ny = 61  # number of grid cells y dir
    dim = (Ny, Nx)
    pdim = Nx * Ny

    #Slice for the rapidity
    Jx = slice(0, Nx)        # 1:Nx
    J1x =  slice(1, Nx-1)     # 2:Nx-1 (interior cells)
    J2x =  slice(0, Nx-1)     # 1:Nx-1 (interfaces)
    J5x =  slice(1, Nx)       # 2:Nx

    Jy =  slice(0, Ny)        # 1:Ny
    J1y =  slice(1, Ny-1)     # 2:Ny-1
    J2y =  slice(0, Ny-1)     # 1:Ny-1
    J5y =  slice(1, Ny)       # 2:Ny

    #########################
    # 2. Setting parameters #
    #########################
    alpC0 = alpha_c.reshape(dim, order='F') + 1e-10
    alpF0 = initialEnsemble[0:pdim].reshape(dim, order='F') + 1e-10
    khat_w = initialEnsemble[pdim:2*pdim].reshape(dim, order='F')
    T_v = initialEnsemble[2*pdim:3*pdim].reshape(dim, order='F')
    T_l = initialEnsemble[3*pdim:4*pdim].reshape(dim, order='F')
    Kgrow = initialEnsemble[4*pdim:5*pdim].reshape(dim, order='F')
    Thetagrow = initialEnsemble[5*pdim:6*pdim].reshape(dim, order='F')

    ############################
    # 2.1 Reference parameters #
    ############################
    T_star = 10000.0        # s
    L_star = 0.01           # m 
    D_star = L_star**2 / T_star
    rho_star = 1.0          # kg/m3
    G_star = 0.0001         # kg/m3 (protease)
    C_star = 0.0001         # kg/m3 (chemokine)
    H_star = 0.0001         # kg/m3 (TGF)
    P_star = 10**4          # Pa

    ###########################
    # 2.2 Physical parameters #
    ###########################
    T_phys = float(T_phys)  # s
    L_phys = 0.01           # m (1 cm)

    D_G_phys = 8e-12        # m2/s
    D_C_phys = 7e-12        # m2/s
    D_H_phys = 8e-12        # m2/s
         
    xi_1_phys = 8e4         # m3/kg
    xi_2_phys = 8e4 * 2     # m3/kg

    Lam0b = 0.0             # Pa, b=chemokine
    Lam1b = 25000.0 / 2.0   # Pa
    Lam0c = 0.0             # Pa    c=TGF
    Lam1c = 25000.0         # Pa

    # Capillary pressure      DeltaP = -gamma*log( delta+(1-ac) )
    PcS = 1000.0            # gamma, Pa
    delta = 0.01            # -
    PfS = 7000.0            # Pa
    exponF = 25.0           # -

    # Global pressure gradient
    patm = 101325.0         # Pa
    Pw_L = patm * np.ones((Ny, 1))   # Pa, no global pressure gradient
    Pw_R = patm * np.ones((Ny, 1))   # Pa
    Pw_B = patm * np.ones((1, Nx))
    Pw_T = patm * np.ones((1, Nx))

    # Interaction coefficients
    Conduc_Tumo = 5e-13             # m^2/Pas
    I_w_tumo = (1.0 / Conduc_Tumo) * D_star   # Pa

    r_c, r_f, r_w, r_cf, r_fc = 0.8, 0.6, 0.0, 0.5, 0.5
    I_w = I_w_tumo
    I_c = 5000.0 * I_w
    I_f = 100.0 * I_w
    I_cf = 1000.0 * I_w

    khat_A, khat_B = 0.7, 10.0

    # Source terms parameters
    lam_11_phys = 0.00005 * 0.2
    lam_12_phys = 0.00005 * 0.2
    lam_13_phys = 0.0
    lam_21_phys, lam_22_phys = 10.0, 1.25e-3
    lam_31_phys, lam_32_phys, lam_33_phys = 2.5e-3, 2.0e-6, 2.0e-6
    G_M_phys = 0.00005
    nua = 1.0
    lam_41_phys, lam_44_phys = 3e-3, 1.0e-4
    C_M_phys, M_C, nub = 0.00003, 0.5, 0.25
    lam_51_phys, lam_52_phys, lam_53_phys, lam_54_phys, lam_55_phys = 0.5e-6, 0.5e-6, 0.5e-6, 5e-5, 5e-3
    H_M_phys, M_H, nuc = 0.00005, 0.5, 0.2
    lam_61_phys, lam_62_phys, lam_63_phys = 0.0, 0.0, 0.0

    CC, HH = 1, 1 # Chemotaxis/TGF on

    ################################
    # 2.3 Dimensionless parameters #
    ################################
    T_val = T_phys / T_star
    Lx = L_phys / L_star
    Ly = L_phys / L_star

    t = 0.0  # Time initialisation for save of the initial conditions
    dt = T_val / NTime
    dx = Lx / Nx
    i_dx = 1.0 / dx
    dy = Ly / Ny
    i_dy = 1.0 / dy
    i_dx2 = 1.0 / dx**2
    i_dy2 = 1.0 / dy**2

    # Define cell centers
    x_coords = 0.5 * dx + np.arange(Nx) * dx
    y_coords = 0.5 * dy + np.arange(Ny) * dy

    lambdax = dt / dx**2
    mux = dt / dx
    lambday = dt / dy**2
    muy = dt / dx # Fidélité au bug MATLAB ligne 211

    # Diffusion of protease G (a in code), chemokine C (b in code) and TGF H (c in code)
    Da_0 = D_G_phys / D_star
    Db_0 = D_C_phys / D_star
    Dc_0 = D_H_phys / D_star

    # Lambda
    xi_1 = xi_1_phys * C_star
    xi_2 = xi_2_phys * C_star

    # Source term (Sc=0)
    lam_11 = lam_11_phys * T_star
    # lam_12 = lam_12_phys * T_star # Not used in this version
    # lam_13 = lam_13_phys * T_star # Not used in this version

    # ECM (rho)
    lam_21 = lam_21_phys * T_star * G_star
    lam_22 = lam_22_phys * T_star
    # lam_23 = lam_23_phys * T_star # Not used in this version
    # lam_24 = lam_24_phys * T_star # Not used in this version
    rho_M = rho_star # MATLAB uses rho_M_phys/rho_star, but rho_M_phys is 1.0

    # Protease, G (=a)
    lam_31 = lam_31_phys * T_star
    lam_32 = (lam_32_phys * T_star) / G_star
    lam_33 = (lam_33_phys * T_star) / G_star
    G_M = G_M_phys / G_star

    # Chemokine, C (=b)
    lam_41 = (lam_41_phys * T_star * G_star * rho_star) / C_star
    # lam_42 = (lam_42_phys * T_star * G_star * rho_star) / C_star # Not used
    # lam_43 = (lam_43_phys * T_star * G_star * rho_star) / C_star # Not used
    lam_44 = lam_44_phys * T_star
    C_M = C_M_phys / C_star

    # TGF, H (=c)
    lam_51 = lam_51_phys * T_star / H_star
    lam_52 = (lam_52_phys * T_star) / H_star 
    lam_53 = (lam_53_phys * T_star) / H_star
    lam_54 = lam_54_phys * T_star
    lam_55 = lam_55_phys * T_star
    H_M = H_M_phys / H_star
    
    # Pre-calculated mobility constants
    ic, iff, iw, icf = I_c, I_f, I_w, I_cf
    rc, rf, rw = r_c, r_f, r_w
    rcf, rfc = r_cf, r_fc
    # Note: khat_c and khat_w change, but others are fixed
    rcv_base = ic / icf
    rfv_base = iff / icf
    rwv_base = iw / icf

    # Source term (Sf=0)
    lam_61 = lam_61_phys * T_star
    lam_62 = lam_62_phys * T_star
    lam_63 = lam_63_phys * T_star

    ##########################
    # 2.4 Initial conditions #
    ##########################
    P_v_star = patm + 6000.0
    P_l_star = patm - 300.0
    Pw_L, Pw_R = patm * np.ones((Ny, 1)), patm * np.ones((Ny, 1))
    Pw_B, Pw_T = patm * np.ones((1, Nx)), patm * np.ones((1, Nx))

    # Water source/sink (Q)
    X, Y = np.meshgrid(x_coords, y_coords)
    pW0 = np.zeros(dim)
    for iy in range(Ny):
        pW0[iy, :] = 1000.0 * np.exp(-10.0 * (x_coords - 0.5)**2 - 10.0 * (y_coords[iy] - 0.5)**2)
    pW_guess = pW0 + patm

    ecm0 = 1.0 - 0.5 * alpC0
    a0 = np.zeros(dim)   # a = G, produced protease from cancer cells
    b0 = np.zeros(dim)   # b = C, dissolved chemokine from ECM caused by protease "a"
    c0 = np.zeros(dim)   # c = H, produced TGF from fibroblasts
    
    #####################
    # 3. Define vectors #
    #####################

    ################################
    # 3.1 Vectors for initial data #
    ################################
    # Mobilities at interfaces (Section 5.1)
    lam_c_half_L = np.zeros(dim); lam_c_half_R = np.zeros(dim)
    lam_c_half_B = np.zeros(dim); lam_c_half_T = np.zeros(dim)
    lam_f_half_L = np.zeros(dim); lam_f_half_R = np.zeros(dim)
    lam_f_half_B = np.zeros(dim); lam_f_half_T = np.zeros(dim)
    lam_w_half_L = np.zeros(dim); lam_w_half_R = np.zeros(dim)
    lam_w_half_B = np.zeros(dim); lam_w_half_T = np.zeros(dim)
    lam_T_half_L = np.zeros(dim); lam_T_half_R = np.zeros(dim)
    lam_T_half_B = np.zeros(dim); lam_T_half_T = np.zeros(dim)

    # Darcy Velocities (Section 5.3 & 5.4)
    # _x vectors are (Ny, Nx+1), _y vectors are (Ny+1, Nx)
    U_C_half_x = np.zeros((Ny, Nx+1)); U_C_half_y = np.zeros((Ny+1, Nx))
    U_F_half_x = np.zeros((Ny, Nx+1)); U_F_half_y = np.zeros((Ny+1, Nx))
    U_W_half_x = np.zeros((Ny, Nx+1)); U_W_half_y = np.zeros((Ny+1, Nx))
    U_T_half_x = np.zeros((Ny, Nx+1)); U_T_half_y = np.zeros((Ny+1, Nx))

    # Interstitial Velocities (Section 5.5)
    uu_C_half_x = np.zeros((Ny, Nx+1)); uu_C_half_y = np.zeros((Ny+1, Nx))
    uu_F_half_x = np.zeros((Ny, Nx+1)); uu_F_half_y = np.zeros((Ny+1, Nx))
    uu_W_half_x = np.zeros((Ny, Nx+1)); uu_W_half_y = np.zeros((Ny+1, Nx))

    # Components of interstitial velocities (1 to 5)
    uu_C_half_1_x = np.zeros((Ny, Nx+1)); uu_C_half_2_x = np.zeros((Ny, Nx+1)); uu_C_half_3_x = np.zeros((Ny, Nx+1))
    uu_C_half_4_x = np.zeros((Ny, Nx+1)); uu_C_half_5_x = np.zeros((Ny, Nx+1))
    uu_C_half_1_y = np.zeros((Ny+1, Nx)); uu_C_half_2_y = np.zeros((Ny+1, Nx)); uu_C_half_3_y = np.zeros((Ny+1, Nx))
    uu_C_half_4_y = np.zeros((Ny+1, Nx)); uu_C_half_5_y = np.zeros((Ny+1, Nx))

    uu_F_half_1_x = np.zeros((Ny, Nx+1)); uu_F_half_2_x = np.zeros((Ny, Nx+1)); uu_F_half_3_x = np.zeros((Ny, Nx+1))
    uu_F_half_4_x = np.zeros((Ny, Nx+1)); uu_F_half_5_x = np.zeros((Ny, Nx+1))
    uu_F_half_1_y = np.zeros((Ny+1, Nx)); uu_F_half_2_y = np.zeros((Ny+1, Nx)); uu_F_half_3_y = np.zeros((Ny+1, Nx))
    uu_F_half_4_y = np.zeros((Ny+1, Nx)); uu_F_half_5_y = np.zeros((Ny+1, Nx))

    uu_W_half_1_x = np.zeros((Ny, Nx+1)); uu_W_half_2_x = np.zeros((Ny, Nx+1)); uu_W_half_3_x = np.zeros((Ny, Nx+1))
    uu_W_half_4_x = np.zeros((Ny, Nx+1)); uu_W_half_5_x = np.zeros((Ny, Nx+1))
    uu_W_half_1_y = np.zeros((Ny+1, Nx)); uu_W_half_2_y = np.zeros((Ny+1, Nx)); uu_W_half_3_y = np.zeros((Ny+1, Nx))
    uu_W_half_4_y = np.zeros((Ny+1, Nx)); uu_W_half_5_y = np.zeros((Ny+1, Nx))

    ###########################
    # 3.2 Matrix for n system #
    ###########################

    SSa_x = None; SSa_y = None
    SSb_x = None; SSb_y = None
    SSc_x = None; SSc_y = None

    Fla = np.zeros((Ny, Nx))
    Flb = np.zeros((Ny, Nx))
    Flc = np.zeros((Ny, Nx))

    ##################################
    # 3.3 Vectors for flow functions #
    ##################################
    frac_flow_C_half = np.zeros((Ny-1, Nx-1))
    frac_flow_C_half_mod = np.zeros((Ny-1, Nx-1))

    frac_flow_F_half = np.zeros((Ny-1, Nx-1))
    frac_flow_F_half_mod = np.zeros((Ny-1, Nx-1))

    frac_flow_W_half = np.zeros((Ny-1, Nx-1))
    frac_flow_W_half_mod = np.zeros((Ny-1, Nx-1))

    h1_half = np.zeros((Ny-1, Nx-1))
    h1_C_half_mod = np.zeros((Ny-1, Nx-1))
    h1_W_half_mod = np.zeros((Ny-1, Nx-1))

    h2_half = np.zeros((Ny-1, Nx-1))
    h2_C_half_mod = np.zeros((Ny-1, Nx-1))
    h2_F_half_mod = np.zeros((Ny-1, Nx-1))

    h3_half = np.zeros((Ny-1, Nx-1))
    h3_F_half_mod = np.zeros((Ny-1, Nx-1))
    h3_W_half_mod = np.zeros((Ny-1, Nx-1))

    #######################
    # 5. Initial solution #
    #######################
    khat = 1 - khat_A * (1 - np.exp(-khat_B * alpF0))
    khat_val = khat.copy()  # ← ajouter cette ligne
    khat_c = khat
    khat_f = 1.0

    ############################
    # 5.1 Initial computations #
    ############################
    
    # Variable initialization for mobilities at interfaces
    lam_c_half_L = np.zeros(dim); lam_c_half_R = np.zeros(dim)
    lam_c_half_B = np.zeros(dim); lam_c_half_T = np.zeros(dim)
    lam_f_half_L = np.zeros(dim); lam_f_half_R = np.zeros(dim)
    lam_f_half_B = np.zeros(dim); lam_f_half_T = np.zeros(dim)
    lam_w_half_L = np.zeros(dim); lam_w_half_R = np.zeros(dim)
    lam_w_half_B = np.zeros(dim); lam_w_half_T = np.zeros(dim)
    lam_T_half_L = np.zeros(dim); lam_T_half_R = np.zeros(dim)
    lam_T_half_B = np.zeros(dim); lam_T_half_T = np.zeros(dim)

    # --- Mobilities ---
    # Lam_c
    lam_c = func_lam_c(alpC0, alpF0, I_c, r_c, khat_c, I_f, r_f, khat_f, I_w, r_w, khat_w, I_cf, r_cf, r_fc)
    lam_c_half_L[:, 0] = 0.5 * (lam_c[:, 0] + lam_c[:, 0])
    lam_c_half_R[:, Nx-1] = 0.5 * (lam_c[:, Nx-1] + lam_c[:, Nx-1])
    lam_c_half_L[:, J5x] = 0.5 * (lam_c[:, J2x] + lam_c[:, J5x])
    lam_c_half_R[:, J2x] = 0.5 * (lam_c[:, J2x] + lam_c[:, J5x])
    lam_c_half_B[0, :] = 0.5 * (lam_c[0, :] + lam_c[0, :])
    lam_c_half_T[Ny-1, :] = 0.5 * (lam_c[Ny-1, :] + lam_c[Ny-1, :])
    lam_c_half_B[J5y, :] = 0.5 * (lam_c[J2y, :] + lam_c[J5y, :])
    lam_c_half_T[J2y, :] = 0.5 * (lam_c[J2y, :] + lam_c[J5y, :])

    # Lam_f
    lam_f = func_lam_f(alpC0, alpF0, I_c, r_c, khat_c, I_f, r_f, khat_f, I_w, r_w, khat_w, I_cf, r_cf, r_fc)
    lam_f_half_L[:, 0] = 0.5 * (lam_f[:, 0] + lam_f[:, 0])
    lam_f_half_R[:, Nx-1] = 0.5 * (lam_f[:, Nx-1] + lam_f[:, Nx-1])
    lam_f_half_L[:, J5x] = 0.5 * (lam_f[:, J2x] + lam_f[:, J5x])
    lam_f_half_R[:, J2x] = 0.5 * (lam_f[:, J2x] + lam_f[:, J5x])
    lam_f_half_B[0, :] = 0.5 * (lam_f[0, :] + lam_f[0, :])
    lam_f_half_T[Ny-1, :] = 0.5 * (lam_f[Ny-1, :] + lam_f[Ny-1, :])
    lam_f_half_B[J5y, :] = 0.5 * (lam_f[J2y, :] + lam_f[J5y, :])
    lam_f_half_T[J2y, :] = 0.5 * (lam_f[J2y, :] + lam_f[J5y, :])

    # Lam_w
    lam_w = func_lam_w(alpC0, alpF0, I_c, r_c, khat_c, I_f, r_f, khat_f, I_w, r_w, khat_w, I_cf, r_cf, r_fc)
    lam_w_half_L[:, 0] = 0.5 * (lam_w[:, 0] + lam_w[:, 0])
    lam_w_half_R[:, Nx-1] = 0.5 * (lam_w[:, Nx-1] + lam_w[:, Nx-1])
    lam_w_half_L[:, J5x] = 0.5 * (lam_w[:, J2x] + lam_w[:, J5x])
    lam_w_half_R[:, J2x] = 0.5 * (lam_w[:, J2x] + lam_w[:, J5x])
    lam_w_half_B[0, :] = 0.5 * (lam_w[0, :] + lam_w[0, :])
    lam_w_half_T[Ny-1, :] = 0.5 * (lam_w[Ny-1, :] + lam_w[Ny-1, :])
    lam_w_half_B[J5y, :] = 0.5 * (lam_w[J2y, :] + lam_w[J5y, :])
    lam_w_half_T[J2y, :] = 0.5 * (lam_w[J2y, :] + lam_w[J5y, :])

    # Lam_T
    lam_T = lam_c + lam_f + lam_w
    lam_T_half_L[:, 0] = 0.5 * (lam_T[:, 0] + lam_T[:, 0])
    lam_T_half_R[:, Nx-1] = 0.5 * (lam_T[:, Nx-1] + lam_T[:, Nx-1])
    lam_T_half_L[:, 1:Nx] = 0.5 * (lam_T[:, 0:Nx-1] + lam_T[:, 1:Nx])
    lam_T_half_R[:, 0:Nx-1] = 0.5 * (lam_T[:, 0:Nx-1] + lam_T[:, 1:Nx])
    lam_T_half_B[0, :] = 0.5 * (lam_T[0, :] + lam_T[0, :])
    lam_T_half_T[Ny-1, :] = 0.5 * (lam_T[Ny-1, :] + lam_T[Ny-1, :])
    lam_T_half_B[1:Ny, :] = 0.5 * (lam_T[0:Ny-1, :] + lam_T[1:Ny, :])
    lam_T_half_T[0:Ny-1, :] = 0.5 * (lam_T[0:Ny-1, :] + lam_T[1:Ny, :])

    # --- Chemotaxis and Potentials ---
    Lamb0b = lambda_chem(b0, Lam0b, Lam1b, xi_1) # chemokine (cells)
    Lamb0c = lambda_TGF(c0, Lam0c, Lam1c, xi_2)  # TGF (fibroblasts)

    Delta_P = Func_DeltaP(1 - alpC0, PcS, delta)
    Delta_PF = Func_DeltaPF(alpF0, PfS, exponF)

    #####################################
    # 5.2 Solution of pressure equation #
    #####################################
    pW0 = solve_pressure_sparse(pW_guess, lam_T_half_L, lam_T_half_R, lam_T_half_B, lam_T_half_T, 
                                Pw_L, Pw_R, Pw_B, Pw_T, P_v_star, P_l_star, T_v, T_l, 
                                i_dx2, i_dy2, Nx, Ny)
    pW_guess = pW0.copy()
    pF0 = Delta_PF + pW0
    pC0 = Delta_P + pW0

    ######################
    # 5.3 Total velocity #
    ######################
    U_T_half_x[:, 0] = -2 * i_dx * lam_T_half_L[:, 0] * (pW0[:, 0] - Pw_L.flatten())
    U_T_half_x[:, Nx] = -2 * i_dx * lam_T_half_R[:, Nx-1] * (Pw_R.flatten() - pW0[:, Nx-1])
    U_T_half_x[:, 1:Nx] = -i_dx * (lam_T_half_R[:, 0:Nx-1] * (pW0[:, 1:Nx] - pW0[:, 0:Nx-1]) +
                                   lam_c_half_R[:, 0:Nx-1] * (Delta_P[:, 1:Nx] - Delta_P[:, 0:Nx-1]) +
                                   CC * lam_c_half_R[:, 0:Nx-1] * (Lamb0b[:, 1:Nx] - Lamb0b[:, 0:Nx-1]) +
                                   lam_f_half_R[:, 0:Nx-1] * (Delta_PF[:, 1:Nx] - Delta_PF[:, 0:Nx-1]) +
                                   HH * lam_f_half_R[:, 0:Nx-1] * (Lamb0c[:, 1:Nx] - Lamb0c[:, 0:Nx-1]))

    U_T_half_y[0, :] = -2 * i_dy * lam_T_half_B[0, :] * (pW0[0, :] - Pw_B.flatten())
    U_T_half_y[Ny, :] = -2 * i_dy * lam_T_half_T[Ny-1, :] * (Pw_T.flatten() - pW0[Ny-1, :])
    U_T_half_y[1:Ny, :] = -i_dy * (lam_T_half_T[0:Ny-1, :] * (pW0[1:Ny, :] - pW0[0:Ny-1, :]) +
                                   lam_c_half_T[0:Ny-1, :] * (Delta_P[1:Ny, :] - Delta_P[0:Ny-1, :]) +
                                   CC * lam_c_half_T[0:Ny-1, :] * (Lamb0b[1:Ny, :] - Lamb0b[0:Ny-1, :]) +
                                   lam_f_half_T[0:Ny-1, :] * (Delta_PF[1:Ny, :] - Delta_PF[0:Ny-1, :]) +
                                   HH * lam_f_half_T[0:Ny-1, :] * (Lamb0c[1:Ny, :] - Lamb0c[0:Ny-1, :]))

    ########################
    # 5.4 Darcy velocities #
    ########################
    frac_flow_C = Func_fc(alpC0, alpF0, I_c, r_c, khat_c, I_f, r_f, khat_f, I_w, r_w, khat_w, I_cf, r_cf, r_fc)
    frac_flow_F = Func_ff(alpC0, alpF0, I_c, r_c, khat_c, I_f, r_f, khat_f, I_w, r_w, khat_w, I_cf, r_cf, r_fc)

    h1 = Func_h1(alpC0, alpF0, I_c, r_c, khat_c, I_f, r_f, khat_f, I_w, r_w, khat_w, I_cf, r_cf, r_fc)
    h2 = Func_h2(alpC0, alpF0, I_c, r_c, khat_c, I_f, r_f, khat_f, I_w, r_w, khat_w, I_cf, r_cf, r_fc)
    h3 = Func_h3(alpC0, alpF0, I_c, r_c, khat_c, I_f, r_f, khat_f, I_w, r_w, khat_w, I_cf, r_cf, r_fc)

    frac_flow_C_half_x = 0.5 * (frac_flow_C[:, 1:Nx] + frac_flow_C[:, 0:Nx-1])
    frac_flow_C_half_y = 0.5 * (frac_flow_C[1:Ny, :] + frac_flow_C[0:Ny-1, :])
    frac_flow_F_half_x = 0.5 * (frac_flow_F[:, 1:Nx] + frac_flow_F[:, 0:Nx-1])
    frac_flow_F_half_y = 0.5 * (frac_flow_F[1:Ny, :] + frac_flow_F[0:Ny-1, :])
    frac_flow_W_half_x = 1.0 - frac_flow_C_half_x - frac_flow_F_half_x
    frac_flow_W_half_y = 1.0 - frac_flow_C_half_y - frac_flow_F_half_y

    h1_half_x = 0.5 * (h1[:, 1:Nx] + h1[:, 0:Nx-1])
    h1_half_y = 0.5 * (h1[1:Ny, :] + h1[0:Ny-1, :])
    h2_half_x = 0.5 * (h2[:, 1:Nx] + h2[:, 0:Nx-1])
    h2_half_y = 0.5 * (h2[1:Ny, :] + h2[0:Ny-1, :])
    h3_half_x = 0.5 * (h3[:, 1:Nx] + h3[:, 0:Nx-1])
    h3_half_y = 0.5 * (h3[1:Ny, :] + h3[0:Ny-1, :])

    # --- Cell Velocity ---
    U_C_half_x[:, 1:Nx] = (U_T_half_x[:, 1:Nx] * frac_flow_C_half_x -
                           i_dx * (h1_half_x + h2_half_x) * (Delta_P[:, 1:Nx] - Delta_P[:, 0:Nx-1]) -
                           CC * i_dx * (h1_half_x + h2_half_x) * (Lamb0b[:, 1:Nx] - Lamb0b[:, 0:Nx-1]) +
                           i_dx * h2_half_x * (Delta_PF[:, 1:Nx] - Delta_PF[:, 0:Nx-1]) +
                           HH * i_dx * h2_half_x * (Lamb0c[:, 1:Nx] - Lamb0c[:, 0:Nx-1]))

    U_C_half_y[1:Ny, :] = (U_T_half_y[1:Ny, :] * frac_flow_C_half_y -
                           i_dy * (h1_half_y + h2_half_y) * (Delta_P[1:Ny, :] - Delta_P[0:Ny-1, :]) -
                           CC * i_dy * (h1_half_y + h2_half_y) * (Lamb0b[1:Ny, :] - Lamb0b[0:Ny-1, :]) +
                           i_dy * h2_half_y * (Delta_PF[1:Ny, :] - Delta_PF[0:Ny-1, :]) +
                           HH * i_dy * h2_half_y * (Lamb0c[1:Ny, :] - Lamb0c[0:Ny-1, :]))

    U_F_half_x[:, 1:Nx] = (U_T_half_x[:, 1:Nx] * frac_flow_F_half_x - 
                           i_dx * h2_half_x * (Delta_P[:, 1:Nx] - Delta_P[:, 0:Nx-1]) +
                           CC * i_dx * h2_half_x * (Lamb0b[:, 1:Nx] - Lamb0b[:, 0:Nx-1]) -
                           i_dx * (h2_half_x + h3_half_x) * (Delta_PF[:, 1:Nx] - Delta_PF[:, 0:Nx-1]) -
                           HH * i_dx * (h2_half_x + h3_half_x) * (Lamb0c[:, 1:Nx] - Lamb0c[:, 0:Nx-1]))

    U_F_half_y[1:Ny, :] = (U_T_half_y[1:Ny, :] * frac_flow_F_half_y +
                           i_dy * h2_half_y * (Delta_P[1:Ny, :] - Delta_P[0:Ny-1, :]) +
                           CC * i_dy * h2_half_y * (Lamb0b[1:Ny, :] - Lamb0b[0:Ny-1, :]) -
                           i_dy * (h2_half_y + h3_half_y) * (Delta_PF[1:Ny, :] - Delta_PF[0:Ny-1, :]) -
                           HH * i_dy * (h2_half_y + h3_half_y) * (Lamb0c[1:Ny, :] - Lamb0c[0:Ny-1, :]))

    U_W_half_x[:, 1:Nx] = (U_T_half_x[:, 1:Nx] * frac_flow_W_half_x +
                           i_dx * h1_half_x * (Delta_P[:, 1:Nx] - Delta_P[:, 0:Nx-1]) +
                           CC * i_dx * h1_half_x * (Lamb0b[:, 1:Nx] - Lamb0b[:, 0:Nx-1]) +
                           i_dx * h3_half_x * (Delta_PF[:, 1:Nx] - Delta_PF[:, 0:Nx-1]) +
                           HH * i_dx * h3_half_x * (Lamb0c[:, 1:Nx] - Lamb0c[:, 0:Nx-1]))

    U_W_half_y[1:Ny, :] = (U_T_half_y[1:Ny, :] * frac_flow_W_half_y +
                           i_dy * h1_half_y * (Delta_P[1:Ny, :] - Delta_P[0:Ny-1, :]) +
                           CC * i_dy * h1_half_y * (Lamb0b[1:Ny, :] - Lamb0b[0:Ny-1, :]) +
                           i_dy * h3_half_y * (Delta_PF[1:Ny, :] - Delta_PF[0:Ny-1, :]) +
                           HH * i_dy * h3_half_y * (Lamb0c[1:Ny, :] - Lamb0c[0:Ny-1, :]))

    ###############################
    # 5.5 Interstitial velocities #
    ###############################
    # Flow functions at cell centers
    frac_flow_C_mod = func_fc_mod(alpC0, alpF0, I_c, r_c, khat_c, I_f, r_f, khat_f, I_w, r_w, khat_w, I_cf, r_cf, r_fc)
    frac_flow_F_mod = func_ff_mod(alpC0, alpF0, I_c, r_c, khat_c, I_f, r_f, khat_f, I_w, r_w, khat_w, I_cf, r_cf, r_fc)
    frac_flow_W_mod = func_fw_mod(alpC0, alpF0, I_c, r_c, khat_c, I_f, r_f, khat_f, I_w, r_w, khat_w, I_cf, r_cf, r_fc)

    h1_C_mod = func_h1_c_mod(alpC0, alpF0, I_c, r_c, khat_c, I_f, r_f, khat_f, I_w, r_w, khat_w, I_cf, r_cf, r_fc)
    h1_W_mod = func_h1_w_mod(alpC0, alpF0, I_c, r_c, khat_c, I_f, r_f, khat_f, I_w, r_w, khat_w, I_cf, r_cf, r_fc)
    h2_C_mod = func_h2_c_mod(alpC0, alpF0, I_c, r_c, khat_c, I_f, r_f, khat_f, I_w, r_w, khat_w, I_cf, r_cf, r_fc)
    h2_F_mod = func_h2_f_mod(alpC0, alpF0, I_c, r_c, khat_c, I_f, r_f, khat_f, I_w, r_w, khat_w, I_cf, r_cf, r_fc)
    h3_F_mod = func_h3_f_mod(alpC0, alpF0, I_c, r_c, khat_c, I_f, r_f, khat_f, I_w, r_w, khat_w, I_cf, r_cf, r_fc)
    h3_W_mod = func_h3_w_mod(alpC0, alpF0, I_c, r_c, khat_c, I_f, r_f, khat_f, I_w, r_w, khat_w, I_cf, r_cf, r_fc)

    # Interface averages (X direction)
    frac_flow_C_half_mod_x = 0.5 * (frac_flow_C_mod[:, 1:Nx] + frac_flow_C_mod[:, 0:Nx-1])
    # Correction: Upwinding pour frac_flow_F_half_mod_x comme en MATLAB
    frac_flow_F_half_mod_x = (0.5 * (1 + np.sign(U_F_half_x[:, 1:Nx])) * frac_flow_F_mod[:, 0:Nx-1] +
                              0.5 * (1 - np.sign(U_F_half_x[:, 1:Nx])) * frac_flow_F_mod[:, 1:Nx])
    # Upwind-like logic for Water (consistent with MATLAB logic B/C)
    frac_flow_W_half_mod_x = (0.5 * (1 + np.sign(U_W_half_x[:, 1:Nx])) * frac_flow_W_mod[:, 0:Nx-1] + 
                              0.5 * (1 - np.sign(U_W_half_x[:, 1:Nx])) * frac_flow_W_mod[:, 1:Nx])

    h1_C_half_mod_x = 0.5 * (h1_C_mod[:, 1:Nx] + h1_C_mod[:, 0:Nx-1])
    h1_W_half_mod_x = 0.5 * (h1_W_mod[:, 1:Nx] + h1_W_mod[:, 0:Nx-1])
    h2_C_half_mod_x = 0.5 * (h2_C_mod[:, 1:Nx] + h2_C_mod[:, 0:Nx-1])
    h2_F_half_mod_x = 0.5 * (h2_F_mod[:, 1:Nx] + h2_F_mod[:, 0:Nx-1])
    h3_F_half_mod_x = 0.5 * (h3_F_mod[:, 1:Nx] + h3_F_mod[:, 0:Nx-1])
    h3_W_half_mod_x = 0.5 * (h3_W_mod[:, 1:Nx] + h3_W_mod[:, 0:Nx-1])

    # Interface averages (Y direction)
    frac_flow_C_half_mod_y = 0.5 * (frac_flow_C_mod[1:Ny, :] + frac_flow_C_mod[0:Ny-1, :]) # Correct
    # Correction: Upwinding for frac_flow_F_half_mod_y like in MATLAB
    frac_flow_F_half_mod_y = (0.5 * (1 + np.sign(U_F_half_y[1:Ny, :])) * frac_flow_F_mod[0:Ny-1, :] +
                              0.5 * (1 - np.sign(U_F_half_y[1:Ny, :])) * frac_flow_F_mod[1:Ny, :])
    frac_flow_W_half_mod_y = (0.5 * (1 + np.sign(U_W_half_y[1:Ny, :])) * frac_flow_W_mod[0:Ny-1, :] + 
                              0.5 * (1 - np.sign(U_W_half_y[1:Ny, :])) * frac_flow_W_mod[1:Ny, :])

    h1_C_half_mod_y = 0.5 * (h1_C_mod[1:Ny, :] + h1_C_mod[0:Ny-1, :])
    h1_W_half_mod_y = 0.5 * (h1_W_mod[1:Ny, :] + h1_W_mod[0:Ny-1, :])
    h2_C_half_mod_y = 0.5 * (h2_C_mod[1:Ny, :] + h2_C_mod[0:Ny-1, :])
    h2_F_half_mod_y = 0.5 * (h2_F_mod[1:Ny, :] + h2_F_mod[0:Ny-1, :])
    h3_F_half_mod_y = 0.5 * (h3_F_mod[1:Ny, :] + h3_F_mod[0:Ny-1, :])
    h3_W_half_mod_y = 0.5 * (h3_W_mod[1:Ny, :] + h3_W_mod[0:Ny-1, :])

    # --- Cell Interstitial Velocity X ---
    uu_C_half_1_x[:, 1:Nx] = U_T_half_x[:, 1:Nx] * frac_flow_C_half_mod_x
    uu_C_half_2_x[:, 1:Nx] = -i_dx * (h1_C_half_mod_x + h2_C_half_mod_x) * (Delta_P[:, 1:Nx] - Delta_P[:, 0:Nx-1])
    uu_C_half_3_x[:, 1:Nx] = -CC * i_dx * (h1_C_half_mod_x + h2_C_half_mod_x) * (Lamb0b[:, 1:Nx] - Lamb0b[:, 0:Nx-1])
    uu_C_half_4_x[:, 1:Nx] = HH * i_dx * h2_C_half_mod_x * (Delta_PF[:, 1:Nx] - Delta_PF[:, 0:Nx-1])
    uu_C_half_5_x[:, 1:Nx] = HH * i_dx * h2_C_half_mod_x * (Lamb0c[:, 1:Nx] - Lamb0c[:, 0:Nx-1])
    uu_C_half_x[:, 1:Nx] = (uu_C_half_1_x[:, 1:Nx] + uu_C_half_2_x[:, 1:Nx] + 
                            uu_C_half_3_x[:, 1:Nx] + uu_C_half_4_x[:, 1:Nx] + uu_C_half_5_x[:, 1:Nx])

    # --- Cell Interstitial Velocity Y ---
    uu_C_half_1_y[1:Ny, :] = U_T_half_y[1:Ny, :] * frac_flow_C_half_mod_y
    uu_C_half_2_y[1:Ny, :] = -i_dy * (h1_C_half_mod_y + h2_C_half_mod_y) * (Delta_P[1:Ny, :] - Delta_P[0:Ny-1, :])
    uu_C_half_3_y[1:Ny, :] = -CC * i_dy * (h1_C_half_mod_y + h2_C_half_mod_y) * (Lamb0b[1:Ny, :] - Lamb0b[0:Ny-1, :])
    uu_C_half_4_y[1:Ny, :] = HH * i_dy * h2_C_half_mod_y * (Delta_PF[1:Ny, :] - Delta_PF[0:Ny-1, :])
    uu_C_half_5_y[1:Ny, :] = HH * i_dy * h2_C_half_mod_y * (Lamb0c[1:Ny, :] - Lamb0c[0:Ny-1, :])
    uu_C_half_y[1:Ny, :] = (uu_C_half_1_y[1:Ny, :] + uu_C_half_2_y[1:Ny, :] + 
                            uu_C_half_3_y[1:Ny, :] + uu_C_half_4_y[1:Ny, :] + uu_C_half_5_y[1:Ny, :])

    # --- Fibroblast Interstitial Velocity X ---
    uu_F_half_1_x[:, 1:Nx] = U_T_half_x[:, 1:Nx] * frac_flow_F_half_mod_x
    uu_F_half_2_x[:, 1:Nx] = i_dx * h2_F_half_mod_x * (Delta_P[:, 1:Nx] - Delta_P[:, 0:Nx-1])
    uu_F_half_3_x[:, 1:Nx] = CC * i_dx * h2_F_half_mod_x * (Lamb0b[:, 1:Nx] - Lamb0b[:, 0:Nx-1])
    uu_F_half_4_x[:, 1:Nx] = -HH * i_dy * (h2_F_half_mod_x + h3_F_half_mod_x) * (Delta_PF[:, 1:Nx] - Delta_PF[:, 0:Nx-1])
    uu_F_half_5_x[:, 1:Nx] = -HH * i_dx * (h2_F_half_mod_x + h3_F_half_mod_x) * (Lamb0c[:, 1:Nx] - Lamb0c[:, 0:Nx-1])
    uu_F_half_x[:, 1:Nx] = (uu_F_half_1_x[:, 1:Nx] + uu_F_half_2_x[:, 1:Nx] + 
                            uu_F_half_3_x[:, 1:Nx] + uu_F_half_4_x[:, 1:Nx] + uu_F_half_5_x[:, 1:Nx])

    # --- Fibroblast Interstitial Velocity Y ---
    uu_F_half_1_y[1:Ny, :] = U_T_half_y[1:Ny, :] * frac_flow_F_half_mod_y
    uu_F_half_2_y[1:Ny, :] = i_dy * h2_F_half_mod_y * (Delta_P[1:Ny, :] - Delta_P[0:Ny-1, :])
    uu_F_half_3_y[1:Ny, :] = CC * i_dy * h2_F_half_mod_y * (Lamb0b[1:Ny, :] - Lamb0b[0:Ny-1, :])
    uu_F_half_4_y[1:Ny, :] = -HH * i_dy * (h2_F_half_mod_y + h3_F_half_mod_y) * (Delta_PF[1:Ny, :] - Delta_PF[0:Ny-1, :])
    uu_F_half_5_y[1:Ny, :] = -HH * i_dy * (h2_F_half_mod_y + h3_F_half_mod_y) * (Lamb0c[1:Ny, :] - Lamb0c[0:Ny-1, :])
    uu_F_half_y[1:Ny, :] = (uu_F_half_1_y[1:Ny, :] + uu_F_half_2_y[1:Ny, :] + 
                            uu_F_half_3_y[1:Ny, :] + uu_F_half_4_y[1:Ny, :] + uu_F_half_5_y[1:Ny, :])

    # --- Water Interstitial Velocity X ---
    uu_W_half_1_x[:, 0] = U_T_half_x[:, 0]
    uu_W_half_1_x[:, Nx] = U_T_half_x[:, Nx]
    uu_W_half_1_x[:, 1:Nx] = U_T_half_x[:, 1:Nx] * frac_flow_W_half_mod_x
    uu_W_half_2_x[:, 1:Nx] = i_dx * h1_W_half_mod_x * (Delta_P[:, 1:Nx] - Delta_P[:, 0:Nx-1])
    uu_W_half_3_x[:, 1:Nx] = CC * i_dx * h1_W_half_mod_x * (Lamb0b[:, 1:Nx] - Lamb0b[:, 0:Nx-1])
    uu_W_half_4_x[:, 1:Nx] = HH * i_dx * h3_W_half_mod_x * (Delta_PF[:, 1:Nx] - Delta_PF[:, 0:Nx-1])
    uu_W_half_5_x[:, 1:Nx] = HH * i_dx * h3_W_half_mod_x * (Lamb0c[:, 1:Nx] - Lamb0c[:, 0:Nx-1])
    uu_W_half_x[:, :] = (uu_W_half_1_x + uu_W_half_2_x + uu_W_half_3_x + uu_W_half_4_x + uu_W_half_5_x)

    # --- Water Interstitial Velocity Y ---
    uu_W_half_1_y[0, :] = U_T_half_y[0, :]
    uu_W_half_1_y[Ny, :] = U_T_half_y[Ny, :]
    uu_W_half_1_y[1:Ny, :] = U_T_half_y[1:Ny, :] * frac_flow_W_half_mod_y
    uu_W_half_2_y[1:Ny, :] = i_dy * h1_W_half_mod_y * (Delta_P[1:Ny, :] - Delta_P[0:Ny-1, :])
    uu_W_half_3_y[1:Ny, :] = CC * i_dy * h1_W_half_mod_y * (Lamb0b[1:Ny, :] - Lamb0b[0:Ny-1, :])
    uu_W_half_4_y[1:Ny, :] = HH * i_dy * h3_W_half_mod_y * (Delta_PF[1:Ny, :] - Delta_PF[0:Ny-1, :])
    uu_W_half_5_y[1:Ny, :] = HH * i_dy * h3_W_half_mod_y * (Lamb0c[1:Ny, :] - Lamb0c[0:Ny-1, :])
    uu_W_half_y[:, :] = (uu_W_half_1_y + uu_W_half_2_y + uu_W_half_3_y + uu_W_half_4_y + uu_W_half_5_y)

    #############################
    # 6. Preparing new solution #
    #############################

    ############################
    # 6.1 Vectors for solution #
    ############################
    a = np.zeros(dim); b = np.zeros(dim); c = np.zeros(dim)
    ecm = np.zeros(dim)
    alpC = np.zeros(dim); alpF = np.zeros(dim); alpW = np.zeros(dim)

    # Flux terms
    Upwind_alpC_half_x = np.zeros((Ny, Nx)); Upwind_alpC_half_y = np.zeros((Ny, Nx))
    Upwind_alpF_half_x = np.zeros((Ny, Nx)); Upwind_alpF_half_y = np.zeros((Ny, Nx))
    Upwind_a_half_x = np.zeros((Ny, Nx)); Upwind_a_half_y = np.zeros((Ny, Nx))
    Upwind_b_half_x = np.zeros((Ny, Nx)); Upwind_b_half_y = np.zeros((Ny, Nx))
    Upwind_c_half_x = np.zeros((Ny, Nx)); Upwind_c_half_y = np.zeros((Ny, Nx))

    ###########################
    # 6.2 Matrix for n system #
    ###########################
    # Tridiagonal matrices for the implicit diffusion step (Crank-Nicolson)
    # (N-2) size because we only solve for the interior points.
    
    # --- Substance a (Protease G) ---
    diag_a_x = (1 + 2 * lambdax * Da_0) * np.ones(Nx - 2)
    off_a_x  = (-lambdax * Da_0) * np.ones(Nx - 3)
    SSa_x = diags([off_a_x, diag_a_x, off_a_x], [-1, 0, 1], format='csc')

    diag_a_y = (1 + 2 * lambday * Da_0) * np.ones(Ny - 2)
    off_a_y  = (-lambday * Da_0) * np.ones(Ny - 3)
    SSa_y = diags([off_a_y, diag_a_y, off_a_y], [-1, 0, 1], format='csc')

    # --- Substance b (Chemokine C) ---
    diag_b_x = (1 + 2 * lambdax * Db_0) * np.ones(Nx - 2)
    off_b_x  = (-lambdax * Db_0) * np.ones(Nx - 3)
    SSb_x = diags([off_b_x, diag_b_x, off_b_x], [-1, 0, 1], format='csc')

    diag_b_y = (1 + 2 * lambday * Db_0) * np.ones(Ny - 2)
    off_b_y  = (-lambday * Db_0) * np.ones(Ny - 3)
    SSb_y = diags([off_b_y, diag_b_y, off_b_y], [-1, 0, 1], format='csc')

    # --- Substance c (TGF H) ---
    diag_c_x = (1 + 2 * lambdax * Dc_0) * np.ones(Nx - 2)
    off_c_x  = (-lambdax * Dc_0) * np.ones(Nx - 3)
    SSc_x = diags([off_c_x, diag_c_x, off_c_x], [-1, 0, 1], format='csc')

    diag_c_y = (1 + 2 * lambday * Dc_0) * np.ones(Ny - 2)
    off_c_y  = (-lambday * Dc_0) * np.ones(Ny - 3)
    SSc_y = diags([off_c_y, diag_c_y, off_c_y], [-1, 0, 1], format='csc')

    # --- Pre-factorization for acceleration ---
    solve_a_x = factorized(SSa_x)
    solve_a_y = factorized(SSa_y)
    solve_b_x = factorized(SSb_x)
    solve_b_y = factorized(SSb_y)
    solve_c_x = factorized(SSc_x)
    solve_c_y = factorized(SSc_y)

    #####################
    # 6.3 Initial state #
    #####################
    a_old, b_old, c_old = a0.copy(), b0.copy(), c0.copy()
    ecm_old = ecm0.copy()
    alpC_old, alpF_old = alpC0.copy(), alpF0.copy()
    pW_old = pW0.copy()
    pW = pW_old.copy()

    # --- PRE-ALLOCATION (Crucial for speed and to avoid NameError) ---
    flux_x = np.zeros((Ny, Nx+1)); flux_y = np.zeros((Ny+1, Nx))
    flux_xf = np.zeros((Ny, Nx+1)); flux_yf = np.zeros((Ny+1, Nx))
    flux_a_x = np.zeros((Ny, Nx+1)); flux_a_y = np.zeros((Ny+1, Nx))
    flux_b_x = np.zeros((Ny, Nx+1)); flux_b_y = np.zeros((Ny+1, Nx))
    flux_c_x = np.zeros((Ny, Nx+1)); flux_c_y = np.zeros((Ny+1, Nx))

    # For the validation, using 1 and for the 100 simulations, using 5.
    pressure_freq = 1
    

    #############################
    # 7. Advancing the solution #
    #############################
    results = {}
    for jj in range(NTime):
        t = (jj + 1) * dt 

        # 7.1 Step 0: dt/2 reaction
        Sc = source_cell_theta(alpC_old, ecm_old, lam_11, Thetagrow)
        Sc = Sc * Kgrow
        Sf = source_fibroblast_new(alpF_old, ecm_old, lam_61, lam_62, lam_63)

        alpC = alpC_old * np.exp(0.5 * dt * Sc)
        alpF = alpF_old * np.exp(0.5 * dt * Sf)

        # Ensure minimum volume fraction
        alpC = np.maximum(alpC, 0.0000001)
        alpF = np.maximum(alpF, 0.0000001)
        alpW = 1.0 - alpC - alpF

        # ODE model for development of ECM
        ecm = (ecm_old + 0.5 * dt * lam_22 * ecm_old * (1 - ecm_old / ecm0)) / (1 + 0.5 * dt * lam_21 * np.maximum(a_old, 0))
        ecm = np.clip(ecm, 0.0, 1.0)

        # Update "old" states before transport
        alpC_old = alpC.copy()
        alpF_old = alpF.copy()
        alpW_old = alpW.copy()
        ecm_old = ecm.copy()

        # 7.2 Step 1: dt transport

        khat_val = np.minimum(1.0 - khat_A * (1.0 - np.exp(-khat_B * alpF)), khat_val)
        
        # Calculation of powers for efficiency
        ac_rc = alpC**rc
        af_rf = alpF**rf
        ac_rc_rcf = alpC**(rc - rcf)
        af_rf_rfc = alpF**(rf - rfc)
        ac_2_rcf = alpC**(2 - rcf)
        af_2_rfc = alpF**(2 - rfc)
        aw_val = 1.00001 - alpC - alpF 
        aw_2_rw = aw_val**(2 - rw)

        rcv = ic * khat_val / icf
        rfv = iff / icf # khat_f = 1.0
        rwv = iw * khat_w / icf

        # Shared denominator for mobilities (lam_c, lam_f, lam_w)
        shared_term = rcv * rfv * ac_rc_rcf * af_rf_rfc
        den_mob = shared_term + rcv * ac_rc + rfv * af_rf + 1e-12
        common_inv_mob = rwv / (iw * khat_w * den_mob)

        lam_c = common_inv_mob * (alpC**2 + rfv * ac_2_rcf * af_rf_rfc + alpC * alpF)
        lam_f = common_inv_mob * (alpF**2 + rcv * af_2_rfc * ac_rc_rcf + alpC * alpF)
        lam_w = (aw_2_rw / (iw * khat_w * den_mob)) * (rcv * ac_rc + shared_term + rfv * af_rf)

        #######################################
        # 7.2.1 Solution of pressure equation #
        #######################################
        lam_c_half_L[:, 0] = 0.5 * (lam_c[:, 0] + lam_c[:, 0])
        lam_c_half_R[:, Nx-1] = 0.5 * (lam_c[:, Nx-1] + lam_c[:, Nx-1])
        lam_c_half_L[:, J5x] = 0.5 * (lam_c[:, J2x] + lam_c[:, J5x])
        lam_c_half_R[:, J2x] = 0.5 * (lam_c[:, J2x] + lam_c[:, J5x])
        lam_c_half_B[0, :] = 0.5 * (lam_c[0, :] + lam_c[0, :])
        lam_c_half_T[Ny-1, :] = 0.5 * (lam_c[Ny-1, :] + lam_c[Ny-1, :])
        lam_c_half_B[J5y, :] = 0.5 * (lam_c[J2y, :] + lam_c[J5y, :])
        lam_c_half_T[J2y, :] = 0.5 * (lam_c[J2y, :] + lam_c[J5y, :])

        lam_f_half_L[:, 0] = 0.5 * (lam_f[:, 0] + lam_f[:, 0])
        lam_f_half_R[:, Nx-1] = 0.5 * (lam_f[:, Nx-1] + lam_f[:, Nx-1])
        lam_f_half_L[:, J5x] = 0.5 * (lam_f[:, J2x] + lam_f[:, J5x])
        lam_f_half_R[:, J2x] = 0.5 * (lam_f[:, J2x] + lam_f[:, J5x])
        lam_f_half_B[0, :] = 0.5 * (lam_f[0, :] + lam_f[0, :])
        lam_f_half_T[Ny-1, :] = 0.5 * (lam_f[Ny-1, :] + lam_f[Ny-1, :])
        lam_f_half_B[J5y, :] = 0.5 * (lam_f[J2y, :] + lam_f[J5y, :])
        lam_f_half_T[J2y, :] = 0.5 * (lam_f[J2y, :] + lam_f[J5y, :])

        lam_w_half_L[:, 0] = 0.5 * (lam_w[:, 0] + lam_w[:, 0])
        lam_w_half_R[:, Nx-1] = 0.5 * (lam_w[:, Nx-1] + lam_w[:, Nx-1])
        lam_w_half_L[:, J5x] = 0.5 * (lam_w[:, J2x] + lam_w[:, J5x])
        lam_w_half_R[:, J2x] = 0.5 * (lam_w[:, J2x] + lam_w[:, J5x])
        lam_w_half_B[0, :] = 0.5 * (lam_w[0, :] + lam_w[0, :])
        lam_w_half_T[Ny-1, :] = 0.5 * (lam_w[Ny-1, :] + lam_w[Ny-1, :])
        lam_w_half_B[J5y, :] = 0.5 * (lam_w[J2y, :] + lam_w[J5y, :])
        lam_w_half_T[J2y, :] = 0.5 * (lam_w[J2y, :] + lam_w[J5y, :])

        lam_T = lam_c + lam_f + lam_w
        lam_T_half_L[:, 0] = 0.5 * (lam_T[:, 0] + lam_T[:, 0])
        lam_T_half_R[:, Nx-1] = 0.5 * (lam_T[:, Nx-1] + lam_T[:, Nx-1])
        lam_T_half_L[:, 1:Nx] = 0.5 * (lam_T[:, 0:Nx-1] + lam_T[:, 1:Nx])
        lam_T_half_R[:, 0:Nx-1] = 0.5 * (lam_T[:, 0:Nx-1] + lam_T[:, 1:Nx])
        lam_T_half_B[0, :] = 0.5 * (lam_T[0, :] + lam_T[0, :])
        lam_T_half_T[Ny-1, :] = 0.5 * (lam_T[Ny-1, :] + lam_T[Ny-1, :])
        lam_T_half_B[1:Ny, :] = 0.5 * (lam_T[0:Ny-1, :] + lam_T[1:Ny, :])
        lam_T_half_T[0:Ny-1, :] = 0.5 * (lam_T[0:Ny-1, :] + lam_T[1:Ny, :])

        # Potentials (Inlined for speed: avoids 10,000 function calls per member)
        Lambb = Lam0b - Lam1b / (1 + np.exp(-xi_1 * (b_old - 0.3)))
        Lambc = Lam0c - Lam1c / (1 + np.exp(-xi_2 * (c_old - 0.2)))
        
        # Capillary pressures (Inlined)
        Delta_P = -PcS * np.log(delta + (1.0 - alpC))
        Delta_PF = PfS * (alpF**exponF)

        if jj % pressure_freq == 0:
            pW = solve_pressure_sparse(pW_guess, lam_T_half_L, lam_T_half_R, lam_T_half_B, lam_T_half_T, 
                                       Pw_L, Pw_R, Pw_B, Pw_T, P_v_star, P_l_star, T_v, T_l, i_dx2, i_dy2, Nx, Ny)
            pW_guess = pW.copy()


        pF = Delta_PF + pW
        pC = Delta_P + pW

        ########################
        # 7.2.2 Total velocity #
        ########################
        U_T_half_x[:, 0] = -2 * i_dx * lam_T_half_L[:, 0] * (pW[:, 0] - Pw_L.flatten())
        U_T_half_x[:, Nx] = -2 * i_dx * lam_T_half_R[:, Nx-1] * (Pw_R.flatten() - pW[:, Nx-1])
        U_T_half_x[:, 1:Nx] = -i_dx * (lam_T_half_R[:, 0:Nx-1] * (pW[:, 1:Nx] - pW[:, 0:Nx-1]) +
                                       lam_c_half_R[:, 0:Nx-1] * (Delta_P[:, 1:Nx] - Delta_P[:, 0:Nx-1]) +
                                       CC * lam_c_half_R[:, 0:Nx-1] * (Lambb[:, 1:Nx] - Lambb[:, 0:Nx-1]) +
                                       lam_f_half_R[:, 0:Nx-1] * (Delta_PF[:, 1:Nx] - Delta_PF[:, 0:Nx-1]) +
                                       HH * lam_f_half_R[:, 0:Nx-1] * (Lambc[:, 1:Nx] - Lambc[:, 0:Nx-1]))

        U_T_half_y[0, :] = -2 * i_dy * lam_T_half_B[0, :] * (pW[0, :] - Pw_B.flatten())
        U_T_half_y[Ny, :] = -2 * i_dy * lam_T_half_T[Ny-1, :] * (Pw_T.flatten() - pW[Ny-1, :])
        U_T_half_y[1:Ny, :] = -i_dy * (lam_T_half_T[0:Ny-1, :] * (pW[1:Ny, :] - pW[0:Ny-1, :]) +
                                       lam_c_half_T[0:Ny-1, :] * (Delta_P[1:Ny, :] - Delta_P[0:Ny-1, :]) +
                                       CC * lam_c_half_T[0:Ny-1, :] * (Lambb[1:Ny, :] - Lambb[0:Ny-1, :]) +
                                       lam_f_half_T[0:Ny-1, :] * (Delta_PF[1:Ny, :] - Delta_PF[0:Ny-1, :]) +
                                       HH * lam_f_half_T[0:Ny-1, :] * (Lambc[1:Ny, :] - Lambc[0:Ny-1, :]))

        ##########################
        # 7.2.3 Darcy velocities #
        ##########################
        den_flow = (alpC + alpF)**2 + rfv * ac_2_rcf * af_rf_rfc + rcv * ac_rc_rcf * af_2_rfc + (aw_val**(2-rw) / rwv) * (den_mob) + 1e-12
        inv_den_flow = 1.0 / den_flow
        frac_flow_C = (alpC * alpF + alpC**2 + rfv * ac_2_rcf * af_rf_rfc) * inv_den_flow
        frac_flow_F = (alpC * alpF + alpF**2 + rcv * ac_rc_rcf * af_2_rfc) * inv_den_flow
        h1 = (aw_val**(2-rw) / (iw * khat_w)) * (rfv * ac_2_rcf * af_rf_rfc + alpC**2 + alpC * alpF) * inv_den_flow
        h2 = (1.0 / (iw * khat_w)) * (rwv * ac_2_rcf * af_2_rfc - alpC * alpF * aw_val**(2-rw)) * inv_den_flow
        h3 = (aw_val**(2-rw) / (iw * khat_w)) * (alpC * alpF + rcv * ac_rc_rcf * af_2_rfc + alpF**2) * inv_den_flow

        frac_flow_C_half_x = 0.5 * (frac_flow_C[:, 1:Nx] + frac_flow_C[:, 0:Nx-1])
        frac_flow_C_half_y = 0.5 * (frac_flow_C[1:Ny, :] + frac_flow_C[0:Ny-1, :])
        frac_flow_F_half_x = 0.5 * (frac_flow_F[:, 1:Nx] + frac_flow_F[:, 0:Nx-1])
        frac_flow_F_half_y = 0.5 * (frac_flow_F[1:Ny, :] + frac_flow_F[0:Ny-1, :])
        frac_flow_W_half_x = 1.0 - frac_flow_C_half_x - frac_flow_F_half_x
        frac_flow_W_half_y = 1.0 - frac_flow_C_half_y - frac_flow_F_half_y

        h1_half_x = 0.5 * (h1[:, 1:Nx] + h1[:, 0:Nx-1])
        h1_half_y = 0.5 * (h1[1:Ny, :] + h1[0:Ny-1, :])
        h2_half_x = 0.5 * (h2[:, 1:Nx] + h2[:, 0:Nx-1])
        h2_half_y = 0.5 * (h2[1:Ny, :] + h2[0:Ny-1, :])
        h3_half_x = 0.5 * (h3[:, 1:Nx] + h3[:, 0:Nx-1])
        h3_half_y = 0.5 * (h3[1:Ny, :] + h3[0:Ny-1, :])

        # Darcy Velocity Updates
        U_C_half_x[:, 1:Nx] = (U_T_half_x[:, 1:Nx] * frac_flow_C_half_x - i_dx * (h1_half_x + h2_half_x) * (Delta_P[:, 1:Nx] - Delta_P[:, 0:Nx-1]) -
                               CC * i_dx * (h1_half_x + h2_half_x) * (Lambb[:, 1:Nx] - Lambb[:, 0:Nx-1]) +
                               i_dx * h2_half_x * (Delta_PF[:, 1:Nx] - Delta_PF[:, 0:Nx-1]) +
                               HH * i_dx * h2_half_x * (Lambc[:, 1:Nx] - Lambc[:, 0:Nx-1]))
        U_C_half_y[1:Ny, :] = (U_T_half_y[1:Ny, :] * frac_flow_C_half_y - i_dy * (h1_half_y + h2_half_y) * (Delta_P[1:Ny, :] - Delta_P[0:Ny-1, :]) -
                               CC * i_dy * (h1_half_y + h2_half_y) * (Lambb[1:Ny, :] - Lambb[0:Ny-1, :]) +
                               i_dy * h2_half_y * (Delta_PF[1:Ny, :] - Delta_PF[0:Ny-1, :]) +
                               HH * i_dy * h2_half_y * (Lambc[1:Ny, :] - Lambc[0:Ny-1, :]))

        U_F_half_x[:, 1:Nx] = (U_T_half_x[:, 1:Nx] * frac_flow_F_half_x + i_dx * h2_half_x * (Delta_P[:, 1:Nx] - Delta_P[:, 0:Nx-1]) +
                               CC * i_dx * h2_half_x * (Lambb[:, 1:Nx] - Lambb[:, 0:Nx-1]) -
                               i_dx * (h2_half_x + h3_half_x) * (Delta_PF[:, 1:Nx] - Delta_PF[:, 0:Nx-1]) -
                               HH * i_dx * (h2_half_x + h3_half_x) * (Lambc[:, 1:Nx] - Lambc[:, 0:Nx-1]))
        U_F_half_y[1:Ny, :] = (U_T_half_y[1:Ny, :] * frac_flow_F_half_y + i_dy * h2_half_y * (Delta_P[1:Ny, :] - Delta_P[0:Ny-1, :]) +
                               CC * i_dy * h2_half_y * (Lambb[1:Ny, :] - Lambb[0:Ny-1, :]) -
                               i_dy * (h2_half_y + h3_half_y) * (Delta_PF[1:Ny, :] - Delta_PF[0:Ny-1, :]) -
                               HH * i_dy * (h2_half_y + h3_half_y) * (Lambc[1:Ny, :] - Lambc[0:Ny-1, :]))
        U_W_half_x[:, 1:Nx] = (U_T_half_x[:, 1:Nx] * frac_flow_W_half_x + i_dx * h1_half_x * (Delta_P[:, 1:Nx] - Delta_P[:, 0:Nx-1]) +
                               CC * i_dx * h1_half_x * (Lambb[:, 1:Nx] - Lambb[:, 0:Nx-1]) +
                               i_dx * h3_half_x * (Delta_PF[:, 1:Nx] - Delta_PF[:, 0:Nx-1]) +
                               HH * i_dx * h3_half_x * (Lambc[:, 1:Nx] - Lambc[:, 0:Nx-1]))
        U_W_half_y[1:Ny, :] = (U_T_half_y[1:Ny, :] * frac_flow_W_half_y + i_dy * h1_half_y * (Delta_P[1:Ny, :] - Delta_P[0:Ny-1, :]) +
                               CC * i_dy * h1_half_y * (Lambb[1:Ny, :] - Lambb[0:Ny-1, :]) +
                               i_dy * h3_half_y * (Delta_PF[1:Ny, :] - Delta_PF[0:Ny-1, :]) +
                               HH * i_dy * h3_half_y * (Lambc[1:Ny, :] - Lambc[0:Ny-1, :]))
        #################################
        # 7.2.4 Interstitial velocities #
        #################################
        # Interstitial velocities mod (Inlined)
        frac_flow_C_mod = (alpF + alpC + rfv * (alpC**(1-rcf)) * af_rf_rfc) * inv_den_flow
        frac_flow_F_mod = (alpC + alpF + rcv * ac_rc_rcf * (alpF**(1-rfc))) * inv_den_flow
        frac_flow_W_mod = (aw_val**(1-rw) / rwv * (den_mob - 1e-12)) * inv_den_flow
        h1_C_mod = (aw_val**(2-rw) / (iw * khat_w * den_flow)) * (rfv * (alpC**(1-rcf)) * af_rf_rfc + alpC + alpF)
        h1_W_mod = (aw_val**(1-rw) / (iw * khat_w * den_flow)) * (rfv * ac_2_rcf * af_rf_rfc + alpC**2 + alpC * alpF)
        h2_C_mod = (1.0 / (iw * khat_w * den_flow)) * (rwv * (alpC**(1-rcf)) * (alpF**(2-rfc)) - alpF * aw_2_rw)
        h2_F_mod = (1.0 / (iw * khat_w * den_flow)) * (rwv * ac_2_rcf * (alpF**(1-rfc)) - alpC * aw_2_rw)
        h3_F_mod = (aw_val**(2-rw) / (iw * khat_w * den_flow)) * (alpC + rcv * ac_rc_rcf * (alpF**(1-rfc)) + alpF)
        h3_W_mod = (aw_val**(1-rw) / (iw * khat_w * den_flow)) * (alpC * alpF + rcv * ac_rc_rcf * af_2_rfc + alpF**2)

        # Moyennage aux interfaces
        frac_flow_C_half_mod_x[:, J2x] = 0.5 * (frac_flow_C_mod[:, J5x] + frac_flow_C_mod[:, J2x])
        frac_flow_C_half_mod_y[J2y, :] = 0.5 * (frac_flow_C_mod[J5y, :] + frac_flow_C_mod[J2y, :])
        frac_flow_F_half_mod_x[:, J2x] = (0.5 * (1 + np.sign(U_F_half_x[:, 1:Nx])) * frac_flow_F_mod[:, J2x] +
                                      0.5 * (1 - np.sign(U_F_half_x[:, 1:Nx])) * frac_flow_F_mod[:, J5x])
        frac_flow_F_half_mod_y[J2y, :] = (0.5 * (1 + np.sign(U_F_half_y[1:Ny, :])) * frac_flow_F_mod[J2y, :] +
                                      0.5 * (1 - np.sign(U_F_half_y[1:Ny, :])) * frac_flow_F_mod[J5y, :])
        frac_flow_W_half_mod_x[:, J2x] = 0.5 * (1 + np.sign(U_W_half_x[:, 1:Nx])) * frac_flow_W_mod[:, J2x] + 0.5 * (1 - np.sign(U_W_half_x[:, 1:Nx])) * frac_flow_W_mod[:, J5x]
        frac_flow_W_half_mod_y[J2y, :] = 0.5 * (1 + np.sign(U_W_half_y[1:Ny, :])) * frac_flow_W_mod[J2y, :] + 0.5 * (1 - np.sign(U_W_half_y[1:Ny, :])) * frac_flow_W_mod[J5y, :]

        h1_C_half_mod_x[:, J2x] = 0.5*(h1_C_mod[:, J5x] + h1_C_mod[:, J2x])
        h1_C_half_mod_y[J2y, :] = 0.5*(h1_C_mod[J5y, :] + h1_C_mod[J2y, :])
        h1_W_half_mod_x[:, J2x] = 0.5*(h1_W_mod[:, J5x] + h1_W_mod[:, J2x])
        h1_W_half_mod_y[J2y, :] = 0.5*(h1_W_mod[J5y, :] + h1_W_mod[J2y, :])
        h2_C_half_mod_x[:, J2x] = 0.5*(h2_C_mod[:, J5x] + h2_C_mod[:, J2x])
        h2_C_half_mod_y[J2y, :] = 0.5*(h2_C_mod[J5y, :] + h2_C_mod[J2y, :])
        h2_F_half_mod_x[:, J2x] = 0.5*(h2_F_mod[:, J5x] + h2_F_mod[:, J2x])
        h2_F_half_mod_y[J2y, :] = 0.5*(h2_F_mod[J5y, :] + h2_F_mod[J2y, :])
        h3_F_half_mod_x[:, J2x] = 0.5*(h3_F_mod[:, J5x] + h3_F_mod[:, J2x])
        h3_F_half_mod_y[J2y, :] = 0.5*(h3_F_mod[J5y, :] + h3_F_mod[J2y, :])
        h3_W_half_mod_x[:, J2x] = 0.5*(h3_W_mod[:, J5x] + h3_W_mod[:, J2x])
        h3_W_half_mod_y[J2y, :] = 0.5*(h3_W_mod[J5y, :] + h3_W_mod[J2y, :])

        # --- Interstitial velocities Cell (C) ---
        uu_C_half_1_x[:, 1:Nx] = U_T_half_x[:, 1:Nx] * frac_flow_C_half_mod_x[:, J2x]
        uu_C_half_2_x[:, 1:Nx] = -i_dx * (h1_C_half_mod_x[:, J2x] + h2_C_half_mod_x[:, J2x]) * (Delta_P[:, J5x] - Delta_P[:, J2x])
        uu_C_half_3_x[:, 1:Nx] = -CC * i_dx * (h1_C_half_mod_x[:, J2x] + h2_C_half_mod_x[:, J2x]) * (Lambb[:, J5x] - Lambb[:, J2x])
        uu_C_half_4_x[:, 1:Nx] = HH * i_dx * h2_C_half_mod_x[:, J2x] * (Delta_PF[:, J5x] - Delta_PF[:, J2x])
        uu_C_half_5_x[:, 1:Nx] = HH * i_dx * h2_C_half_mod_x[:, J2x] * (Lambc[:, J5x] - Lambc[:, J2x])
        uu_C_half_x[:, 1:Nx] = uu_C_half_1_x[:, 1:Nx] + uu_C_half_2_x[:, 1:Nx] + uu_C_half_3_x[:, 1:Nx] + uu_C_half_4_x[:, 1:Nx] + uu_C_half_5_x[:, 1:Nx]

        uu_C_half_1_y[1:Ny, :] = U_T_half_y[1:Ny, :] * frac_flow_C_half_mod_y[J2y, :]
        uu_C_half_2_y[1:Ny, :] = -i_dy * (h1_C_half_mod_y[J2y, :] + h2_C_half_mod_y[J2y, :]) * (Delta_P[J5y, :] - Delta_P[J2y, :])
        uu_C_half_3_y[1:Ny, :] = -CC * i_dy * (h1_C_half_mod_y[J2y, :] + h2_C_half_mod_y[J2y, :]) * (Lambb[J5y, :] - Lambb[J2y, :])
        uu_C_half_4_y[1:Ny, :] = HH * i_dy * h2_C_half_mod_y[J2y, :] * (Delta_PF[J5y, :] - Delta_PF[J2y, :])
        uu_C_half_5_y[1:Ny, :] = HH * i_dy * h2_C_half_mod_y[J2y, :] * (Lambc[J5y, :] - Lambc[J2y, :])
        uu_C_half_y[1:Ny, :] = uu_C_half_1_y[1:Ny, :] + uu_C_half_2_y[1:Ny, :] + uu_C_half_3_y[1:Ny, :] + uu_C_half_4_y[1:Ny, :] + uu_C_half_5_y[1:Ny, :]

        # --- Interstitial velocities Fibroblast (F) ---
        uu_F_half_1_x[:, 1:Nx] = U_T_half_x[:, 1:Nx] * frac_flow_F_half_mod_x[:, J2x]
        uu_F_half_2_x[:, 1:Nx] = i_dx * h2_F_half_mod_x[:, J2x] * (Delta_P[:, J5x] - Delta_P[:, J2x])
        uu_F_half_3_x[:, 1:Nx] = CC * i_dx * h2_F_half_mod_x[:, J2x] * (Lambb[:, J5x] - Lambb[:, J2x])
        uu_F_half_4_x[:, 1:Nx] = -HH * i_dx * (h2_F_half_mod_x[:, J2x] + h3_F_half_mod_x[:, J2x]) * (Delta_PF[:, J5x] - Delta_PF[:, J2x])
        uu_F_half_5_x[:, 1:Nx] = -HH * i_dx * (h2_F_half_mod_x[:, J2x] + h3_F_half_mod_x[:, J2x]) * (Lambc[:, J5x] - Lambc[:, J2x])
        uu_F_half_x[:, 1:Nx] = uu_F_half_1_x[:, 1:Nx] + uu_F_half_2_x[:, 1:Nx] + uu_F_half_3_x[:, 1:Nx] + uu_F_half_4_x[:, 1:Nx] + uu_F_half_5_x[:, 1:Nx]

        uu_F_half_1_y[1:Ny, :] = U_T_half_y[1:Ny, :] * frac_flow_F_half_mod_y[J2y, :]
        uu_F_half_2_y[1:Ny, :] = i_dy * h2_F_half_mod_y[J2y, :] * (Delta_P[J5y, :] - Delta_P[J2y, :])
        uu_F_half_3_y[1:Ny, :] = CC * i_dy * h2_F_half_mod_y[J2y, :] * (Lambb[J5y, :] - Lambb[J2y, :])
        uu_F_half_4_y[1:Ny, :] = -HH * i_dy * (h2_F_half_mod_y[J2y, :] + h3_F_half_mod_y[J2y, :]) * (Delta_PF[J5y, :] - Delta_PF[J2y, :])
        uu_F_half_5_y[1:Ny, :] = -HH * i_dy * (h2_F_half_mod_y[J2y, :] + h3_F_half_mod_y[J2y, :]) * (Lambc[J5y, :] - Lambc[J2y, :])
        uu_F_half_y[1:Ny, :] = uu_F_half_1_y[1:Ny, :] + uu_F_half_2_y[1:Ny, :] + uu_F_half_3_y[1:Ny, :] + uu_F_half_4_y[1:Ny, :] + uu_F_half_5_y[1:Ny, :]

        # --- Interstitial velocities Water (W) ---
        uu_W_half_1_x[:, 0], uu_W_half_1_x[:, Nx] = U_T_half_x[:, 0], U_T_half_x[:, Nx]
        uu_W_half_1_x[:, 1:Nx] = U_T_half_x[:, 1:Nx] * frac_flow_W_half_mod_x[:, J2x]
        uu_W_half_2_x[:, 1:Nx] = i_dx * h1_W_half_mod_x[:, J2x] * (Delta_P[:, J5x] - Delta_P[:, J2x])
        uu_W_half_3_x[:, 1:Nx] = CC * i_dx * h1_W_half_mod_x[:, J2x] * (Lambb[:, J5x] - Lambb[:, J2x])
        uu_W_half_4_x[:, 1:Nx] = HH * i_dx * h3_W_half_mod_x[:, J2x] * (Delta_PF[:, J5x] - Delta_PF[:, J2x])
        uu_W_half_5_x[:, 1:Nx] = HH * i_dx * h3_W_half_mod_x[:, J2x] * (Lambc[:, J5x] - Lambc[:, J2x])
        uu_W_half_x[:, :] = uu_W_half_1_x + uu_W_half_2_x + uu_W_half_3_x + uu_W_half_4_x + uu_W_half_5_x

        uu_W_half_1_y[0, :], uu_W_half_1_y[Ny, :] = U_T_half_y[0, :], U_T_half_y[Ny, :]
        uu_W_half_1_y[1:Ny, :] = U_T_half_y[1:Ny, :] * frac_flow_W_half_mod_y[J2y, :]
        uu_W_half_2_y[1:Ny, :] = i_dy * h1_W_half_mod_y[J2y, :] * (Delta_P[J5y, :] - Delta_P[J2y, :])
        uu_W_half_3_y[1:Ny, :] = CC * i_dy * h1_W_half_mod_y[J2y, :] * (Lambb[J5y, :] - Lambb[J2y, :])
        uu_W_half_4_y[1:Ny, :] = HH * i_dy * h3_W_half_mod_y[J2y, :] * (Delta_PF[J5y, :] - Delta_PF[J2y, :])
        uu_W_half_5_y[1:Ny, :] = HH * i_dy * h3_W_half_mod_y[J2y, :] * (Lambc[J5y, :] - Lambc[J2y, :])
        uu_W_half_y[:, :] = uu_W_half_1_y + uu_W_half_2_y + uu_W_half_3_y + uu_W_half_4_y + uu_W_half_5_y


        ##########################
        # 7.2.5 Numerical scheme #
        ##########################
        # --- Upwind fluxes for phases ---
        Upwind_alpC_half_x[:, J2x] = 0.5*(alpC_old[:, J5x] + alpC_old[:, J2x])*uu_C_half_x[:, 1:Nx] - 0.5*np.abs(uu_C_half_x[:, 1:Nx])*(alpC_old[:, J5x] - alpC_old[:, J2x])
        Upwind_alpC_half_y[J2y, :] = 0.5*(alpC_old[J5y, :] + alpC_old[J2y, :])*uu_C_half_y[1:Ny, :] - 0.5*np.abs(uu_C_half_y[1:Ny, :])*(alpC_old[J5y, :] - alpC_old[J2y, :])

        Upwind_alpF_half_x[:, J2x] = 0.5*(alpF_old[:, J5x] + alpF_old[:, J2x])*uu_F_half_x[:, 1:Nx] - 0.5*np.abs(uu_F_half_x[:, 1:Nx])*(alpF_old[:, J5x] - alpF_old[:, J2x])
        Upwind_alpF_half_y[J2y, :] = 0.5*(alpF_old[J5y, :] + alpF_old[J2y, :])*uu_F_half_y[1:Ny, :] - 0.5*np.abs(uu_F_half_y[1:Ny, :])*(alpF_old[J5y, :] - alpF_old[J2y, :])

        # --- Update alpC (Volume Fraction) ---
        # Vectorized Divergence Calculation (Replaces Coin/Bord logic)
        flux_x = np.zeros((Ny, Nx+1))
        flux_x[:, 1:Nx] = Upwind_alpC_half_x[:, 0:Nx-1]
        flux_y = np.zeros((Ny+1, Nx))
        flux_y[1:Ny, :] = Upwind_alpC_half_y[0:Ny-1, :]
        
        alpC = alpC_old - mux * np.diff(flux_x, axis=1) - muy * np.diff(flux_y, axis=0)

        # --- Update alpF (Volume Fraction) ---
        flux_xf = np.zeros((Ny, Nx+1))
        flux_xf[:, 1:Nx] = Upwind_alpF_half_x[:, 0:Nx-1]
        flux_yf = np.zeros((Ny+1, Nx))
        flux_yf[1:Ny, :] = Upwind_alpF_half_y[0:Ny-1, :]

        alpF = alpF_old - mux * np.diff(flux_xf, axis=1) - muy * np.diff(flux_yf, axis=0)

        # Clip et Volume de l'eau
        alpC = np.maximum(alpC, 0.0000001)
        alpF = np.maximum(alpF, 0.0000001)
        alpC = np.minimum(alpC, 0.7)
        alpF = np.minimum(alpF, 0.7)
        alpW = 1.0 - alpC - alpF

        # Final state update before chemical diffusion
        alpC_old = alpC.copy()
        alpF_old = alpF.copy()


        # --- Solve G (Protease) ---
        # Upwind fluxes for G
        Upwind_a_half_x[:, J2x] = 0.5*(a_old[:, J5x] + a_old[:, J2x])*uu_W_half_x[:, 1:Nx] - 0.5*np.abs(uu_W_half_x[:, 1:Nx])*(a_old[:, J5x] - a_old[:, J2x])
        Upwind_a_half_y[J2y, :] = 0.5*(a_old[J5y, :] + a_old[J2y, :])*uu_W_half_y[1:Ny, :] - 0.5*np.abs(uu_W_half_y[1:Ny, :])*(a_old[J5y, :] - a_old[J2y, :])
        
        # Calculate Fla (advection term for G)
        flux_a_x = np.zeros((Ny, Nx+1)); flux_a_x[:, 1:] = Upwind_a_half_x
        flux_a_y = np.zeros((Ny+1, Nx)); flux_a_y[1:, :] = Upwind_a_half_y
        Fla = -dt/dx * np.diff(flux_a_x, axis=1) - dt/dy * np.diff(flux_a_y, axis=0)

        # Vectorized ADI X-sweep for G (Resolves all rows at once)
        a_safe = np.clip(a_old, 0, 10.0)
        rhs_x = a_safe[:, J1x] + 0.5*dt*((alpC[:, J1x]+alpF[:, J1x])*(lam_32 - lam_33*(a_safe[:, J1x]/G_M)**nua) - lam_31*a_safe[:, J1x]) + 0.5*Fla[:, J1x]
        a[:, J1x] = solve_a_x(rhs_x.T).T
        a[:, 0], a[:, Nx-1] = a[:, 1], a[:, Nx-2]
        a = np.clip(a, 0, 10.0)

        a_old = a.copy() # Update for Y-sweep
        
        # Vectorized ADI Y-sweep for G (Resolves all columns at once)
        rhs_y = a_old[J1y, :] + 0.5*dt*((alpC[J1y, :]+alpF[J1y, :])*(lam_32 - lam_33*(a_old[J1y, :]/G_M)**nua) - lam_31*a_old[J1y, :]) + 0.5*Fla[J1y, :]
        a[J1y, :] = solve_a_y(rhs_y)
        a[0, :], a[Ny-1, :] = a[1, :], a[Ny-2, :]
        a_old = np.clip(a, 0, 10.0)

        # --- Solve C (Chemokine) ---
        # Upwind fluxes for C
        Upwind_b_half_x[:, J2x] = 0.5*(b_old[:, J5x] + b_old[:, J2x])*uu_W_half_x[:, 1:Nx] - 0.5*np.abs(uu_W_half_x[:, 1:Nx])*(b_old[:, J5x] - b_old[:, J2x])
        Upwind_b_half_y[J2y, :] = 0.5*(b_old[J5y, :] + b_old[J2y, :])*uu_W_half_y[1:Ny, :] - 0.5*np.abs(uu_W_half_y[1:Ny, :])*(b_old[J5y, :] - b_old[J2y, :])

        # Calculate Flb (advection term for C)
        flux_b_x = np.zeros((Ny, Nx+1)); flux_b_x[:, 1:] = Upwind_b_half_x
        flux_b_y = np.zeros((Ny+1, Nx)); flux_b_y[1:, :] = Upwind_b_half_y
        Flb = -dt/dx * np.diff(flux_b_x, axis=1) - dt/dy * np.diff(flux_b_y, axis=0)

        # Vectorized ADI X-sweep for C
        b_safe = np.clip(b_old, 0, 10.0)
        rhs_x = b_safe[:, J1x] + 0.5*dt*(alpC[:, J1x]*alpF[:, J1x]*lam_41*(1.0 - (b_safe[:, J1x]/C_M)**nub) - b_safe[:, J1x]*T_l[:, J1x]*(pW[:, J1x]-P_l_star)*M_C - lam_44*alpC[:, J1x]*b_safe[:, J1x]) + 0.5*Flb[:, J1x]
        b[:, J1x] = solve_b_x(rhs_x.T).T
        b[:, 0], b[:, Nx-1] = b[:, 1], b[:, Nx-2]
        b = np.clip(b, 0, 10.0)

        b_old = b.copy()
        # Vectorized ADI Y-sweep for C
        rhs_y = b_old[J1y, :] + 0.5*dt*(alpC[J1y, :]*alpF[J1y, :]*lam_41*(1.0 - (b_old[J1y, :]/C_M)**nub) - b_old[J1y, :]*T_l[J1y, :]*(pW[J1y, :]-P_l_star)*M_C - lam_44*alpC[J1y, :]*b_old[J1y, :]) + 0.5*Flb[J1y, :]
        b[J1y, :] = solve_b_y(rhs_y)
        b[0, :], b[Ny-1, :] = b[1, :], b[Ny-2, :]
        b_old = np.clip(b, 0, 10.0)

        # --- Solve H (TGF) ---
        # Upwind fluxes for H
        Upwind_c_half_x[:, J2x] = 0.5*(c_old[:, J5x] + c_old[:, J2x])*uu_W_half_x[:, 1:Nx] - 0.5*np.abs(uu_W_half_x[:, 1:Nx])*(c_old[:, J5x] - c_old[:, J2x])
        Upwind_c_half_y[J2y, :] = 0.5*(c_old[J5y, :] + c_old[J2y, :])*uu_W_half_y[1:Ny, :] - 0.5*np.abs(uu_W_half_y[1:Ny, :])*(c_old[J5y, :] - c_old[J2y, :])

        # Calculate Flc (advection term for H)
        flux_c_x = np.zeros((Ny, Nx+1)); flux_c_x[:, 1:] = Upwind_c_half_x
        flux_c_y = np.zeros((Ny+1, Nx)); flux_c_y[1:, :] = Upwind_c_half_y
        Flc = -dt/dx * np.diff(flux_c_x, axis=1) - dt/dy * np.diff(flux_c_y, axis=0)

        # Vectorized ADI X-sweep for H
        c_safe = np.clip(c_old, 0, 10.0)
        rhs_x = c_safe[:, J1x] + 0.5*dt*(-lam_54*c_safe[:, J1x] - lam_55*c_safe[:, J1x]*alpF[:, J1x] + alpF[:, J1x]*(lam_51 - lam_52*(c_safe[:, J1x]/H_M)**2 - lam_53*(c_safe[:, J1x]/H_M)**nuc)) - 0.5*dt*(T_l[:, J1x]*(pW[:, J1x]-P_l_star))*c_safe[:, J1x]*M_H + 0.5*Flc[:, J1x]
        c[:, J1x] = solve_c_x(rhs_x.T).T # Traitement par blocs pour la rapidité
        c[:, 0], c[:, Nx-1] = c[:, 1], c[:, Nx-2]
        c = np.clip(c, 0, 10.0)
        
        c_old = c.copy()
        # Vectorized ADI Y-sweep for H
        rhs_y = c_old[J1y, :] + 0.5*dt*(-lam_54*c_old[J1y, :] - lam_55*c_old[J1y, :] * alpF[J1y, :] + alpF[J1y, :] * (lam_51 - lam_52 * (c_old[J1y, :]/H_M)**2 - lam_53 * (c_old[J1y, :]/H_M)**nuc)) - 0.5*dt*(T_l[J1y, :] * (pW[J1y, :] - P_l_star)) * c_old[J1y, :] * M_H + 0.5*Flc[J1y, :]
        c[J1y, :] = solve_c_y(rhs_y) # Traitement par blocs
        c[0, :], c[Ny-1, :] = c[1, :], c[Ny-2, :]
        c_old = np.clip(c, 0, 10.0)

        # 7.3 Step 2: dt/2 reaction
        # Re-using alpC_old/alpF_old calculated during chemical diffusion
        Sc = source_cell_theta(alpC_old, ecm_old, lam_11, Thetagrow)
        Sc = Sc * Kgrow
        Sf = source_fibroblast_new(alpF_old, ecm_old, lam_61, lam_62, lam_63)

        alpC = alpC_old * np.exp(0.5 * dt * Sc)
        alpF = alpF_old * np.exp(0.5 * dt * Sf)
        
        alpC = np.maximum(alpC, 0.0000001)
        alpF = np.maximum(alpF, 0.0000001)
        alpC = np.minimum(alpC, 0.7)
        alpF = np.minimum(alpF, 0.7)
        alpW = 1.0 - alpC - alpF
        ecm = (ecm_old + 0.5 * dt * lam_22 * ecm_old * (1 - ecm_old / ecm0)) / (1 + 0.5 * dt * lam_21 * np.maximum(a_old, 0))
        ecm = np.clip(ecm, 0.0, 1.0)
        
        alpC_old, alpF_old, ecm_old = alpC.copy(), alpF.copy(), ecm.copy()

        if capture_half and jj == int(NTime / 2) - 1:
            results['pred_alpha_c_half'] = alpC.flatten(order='F')
            results['pressureHalf'] = pW.flatten(order='F')

            if extended_outputs:
                results['pred_alpha_f_half'] = alpF.flatten(order='F')
                uu_W_x = 0.5 * (uu_W_half_x[:, 1:Nx+1] + uu_W_half_x[:, 0:Nx])
                uu_W_y = 0.5 * (uu_W_half_y[1:Ny+1, :] + uu_W_half_y[0:Ny, :])
                results['pred_uu_W_x_half'] = uu_W_x.flatten(order='F')
                results['pred_uu_W_y_half'] = uu_W_y.flatten(order='F')

    results['pred_alpha_c_full'] = alpC.flatten(order='F')
    results['pressureEnd'] = pW.flatten(order='F')

    if extended_outputs:
        results['pred_alpha_f_full'] = alpF.flatten(order='F')
        uu_W_x = 0.5 * (uu_W_half_x[:, 1:Nx+1] + uu_W_half_x[:, 0:Nx])
        uu_W_y = 0.5 * (uu_W_half_y[1:Ny+1, :] + uu_W_half_y[0:Ny, :])
        results['pred_uu_W_x_full'] = uu_W_x.flatten(order='F')
        results['pred_uu_W_y_full'] = uu_W_y.flatten(order='F')

    return results


# =====================================================================
# Public API Functions (from MATLAB file start)
# =====================================================================

def three_phase_simulator_compartment_full(alpha_c, initialEnsemble, Inc_dt):
    """Simulator A: Equivalent of three_phase_simulator_compartment_full.m"""
    NTime = 5000 + Inc_dt * 1000
    r = _three_phase_simulator_compartment_core(
        alpha_c, initialEnsemble, NTime=NTime, T_phys=500000.0,
        capture_half=True, extended_outputs=False)
    return r['pred_alpha_c_half'], r['pred_alpha_c_full'], r['pressureHalf'], r['pressureEnd']


def three_phase_simulator_compartment(alpha_c, initialEnsemble, Inc_dt):
    """Simulator B: Equivalent of three_phase_simulator_compartment.m"""
    NTime = (5000 + Inc_dt * 1000) / 2
    r = _three_phase_simulator_compartment_core(
        alpha_c, initialEnsemble, NTime=NTime, T_phys=250000.0,
        capture_half=False, extended_outputs=False)
    return r['pred_alpha_c_full'], r['pressureEnd']


def three_phase_simulator_compartment_forecast(alpha_c, initialEnsemble):
    """Simulator C: Equivalent of three_phase_simulator_compartment_forecast.m"""
    r = _three_phase_simulator_compartment_core(
        alpha_c, initialEnsemble, NTime=7000, T_phys=500000.0,
        capture_half=True, extended_outputs=True)
    return (r['pred_alpha_c_half'], r['pred_alpha_c_full'],
            r['pressureHalf'], r['pressureEnd'],
            r['pred_alpha_f_half'], r['pred_alpha_f_full'],
            r['pred_uu_W_x_half'], r['pred_uu_W_y_half'],
            r['pred_uu_W_x_full'], r['pred_uu_W_y_full'])


# --- Helper Functions (from MATLAB file end) ---

def Func_DeltaP(u, PcS, delta):
    return -PcS * np.log(delta + u)

def Func_DeltaPF(u, PfS, expon):
    return PfS * u**expon

def func_lam_c(ac, af, ic, rc, khatc, iff, rf, khatf, iw, rw, khatw, icf, rcf, rfc):
    rcv, rfv, rwv = ic*khatc/icf, iff*khatf/icf, iw*khatw/icf
    num = rwv/(iw*khatw)*(ac**2 + rfv*ac**(2-rcf)*af**(rf-rfc) + ac*af)
    den = rcv*rfv*ac**(rc-rcf)*af**(rf-rfc) + rcv*ac**rc + rfv*af**rf
    return num / (den + 1e-12)
def func_lam_f(ac, af, ic, rc, khatc, iff, rf, khatf, iw, rw, khatw, icf, rcf, rfc):
    rcv, rfv, rwv = ic*khatc/icf, iff*khatf/icf, iw*khatw/icf
    num = rwv/(iw*khatw)*(af**2 + rcv*af**(2-rfc)*ac**(rc-rcf) + ac*af)
    den = rcv*rfv*ac**(rc-rcf)*af**(rf-rfc) + rcv*ac**rc + rfv*af**rf
    return num / (den + 1e-12)
def func_lam_w(ac, af, ic, rc, khatc, iff, rf, khatf, iw, rw, khatw, icf, rcf, rfc):
    aw, rcv, rfv = 1.00001-ac-af, ic*khatc/icf, iff*khatf/icf
    num = aw**(2-rw)/(iw*khatw)*(rcv*ac**rc + rcv*rfv*ac**(rc-rcf)*af**(rf-rfc) + rfv*af**rf)
    den = rcv*rfv*ac**(rc-rcf)*af**(rf-rfc) + rcv*ac**rc + rfv*af**rf
    return num / (den + 1e-12)
def func_fc_mod(ac, af, ic, rc, khatc, iff, rf, khatf, iw, rw, khatw, icf, rcf, rfc):
    aw, rcv, rfv, rwv = 1.00001-ac-af, ic*khatc/icf, iff*khatf/icf, iw*khatw/icf
    num = af + ac + rfv*ac**(1-rcf)*af**(rf-rfc)
    den = (ac+af)**2 + rfv*ac**(2-rcf)*af**(rf-rfc) + rcv*ac**(rc-rcf)*af**(2-rfc) + aw**(2-rw)/rwv*(rcv*ac**rc + rcv*rfv*ac**(rc-rcf)*af**(rf-rfc) + rfv*af**rf)
    return num / (den + 1e-12)
def Func_fc(ac, af, ic, rc, khatc, iff, rf, khatf, iw, rw, khatw, icf, rcf, rfc):
    aw = 1.00001 - ac - af
    rcv, rfv, rwv = ic*khatc/icf, iff*khatf/icf, iw*khatw/icf
    num = ac*af + ac**2 + rfv*ac**(2-rcf)*af**(rf-rfc)
    den = (ac+af)**2 + rfv*ac**(2-rcf)*af**(rf-rfc) + rcv*ac**(rc-rcf)*af**(2-rfc) + aw**(2-rw)/rwv*(rcv*ac**rc + rcv*rfv*ac**(rc-rcf)*af**(rf-rfc) + rfv*af**rf)
    return num / (den + 1e-12)
def Func_ff(ac, af, ic, rc, khatc, iff, rf, khatf, iw, rw, khatw, icf, rcf, rfc):
    aw = 1.00001 - ac - af
    rcv, rfv, rwv = ic*khatc/icf, iff*khatf/icf, iw*khatw/icf
    num = ac*af + af**2 + rcv*ac**(rc-rcf)*af**(2-rfc)
    den = (ac+af)**2 + rfv*ac**(2-rcf)*af**(rf-rfc) + rcv*ac**(rc-rcf)*af**(2-rfc) + aw**(2-rw)/rwv*(rcv*ac**rc + rcv*rfv*ac**(rc-rcf)*af**(rf-rfc) + rfv*af**rf)
    return num / (den + 1e-12)
def Func_fw(ac, af, ic, rc, khatc, iff, rf, khatf, iw, rw, khatw, icf, rcf, rfc):
    aw = 1.00001 - ac - af
    rcv, rfv, rwv = ic*khatc/icf, iff*khatf/icf, iw*khatw/icf
    num = aw**(2-rw)/rwv*(rcv*ac**rc + rcv*rfv*ac**(rc-rcf)*af**(rf-rfc) + rfv*af**rf)
    den = (ac+af)**2 + rfv*ac**(2-rcf)*af**(rf-rfc) + rcv*ac**(rc-rcf)*af**(2-rfc) + aw**(2-rw)/rwv*(rcv*ac**rc + rcv*rfv*ac**(rc-rcf)*af**(rf-rfc) + rfv*af**rf)
    return num / (den + 1e-12)
def Func_h1(ac, af, ic, rc, khatc, iff, rf, khatf, iw, rw, khatw, icf, rcf, rfc):
    aw = 1.00001 - ac - af
    rcv, rfv, rwv = ic*khatc/icf, iff*khatf/icf, iw*khatw/icf
    num = aw**(2-rw)/(iw*khatw)*(rfv*ac**(2-rcf)*af**(rf-rfc) + ac**2 + ac*af)
    den = (ac+af)**2 + rfv*ac**(2-rcf)*af**(rf-rfc) + rcv*ac**(rc-rcf)*af**(2-rfc) + aw**(2-rw)/rwv*(rcv*ac**rc + rcv*rfv*ac**(rc-rcf)*af**(rf-rfc) + rfv*af**rf)
    return num / (den + 1e-12)
def func_h1_c_mod(ac, af, ic, rc, khatc, iff, rf, khatf, iw, rw, khatw, icf, rcf, rfc):
    aw = 1.00001 - ac - af
    rcv, rfv, rwv = ic*khatc/icf, iff*khatf/icf, iw*khatw/icf
    num = aw**(2-rw)/(iw*khatw)*(rfv*ac**(1-rcf)*af**(rf-rfc) + ac + af)
    den = (ac+af)**2 + rfv*ac**(2-rcf)*af**(rf-rfc) + rcv*ac**(rc-rcf)*af**(2-rfc) + aw**(2-rw)/rwv*(rcv*ac**rc + rcv*rfv*ac**(rc-rcf)*af**(rf-rfc) + rfv*af**rf)
    return num / (den + 1e-12)
def func_h1_w_mod(ac, af, ic, rc, khatc, iff, rf, khatf, iw, rw, khatw, icf, rcf, rfc):
    aw = 1.00001 - ac - af
    rcv, rfv, rwv = ic*khatc/icf, iff*khatf/icf, iw*khatw/icf
    num = aw**(1-rw)/(iw*khatw)*(rfv*ac**(2-rcf)*af**(rf-rfc) + ac**2 + ac*af)
    den = (ac+af)**2 + rfv*ac**(2-rcf)*af**(rf-rfc) + rcv*ac**(rc-rcf)*af**(2-rfc) + aw**(2-rw)/rwv*(rcv*ac**rc + rcv*rfv*ac**(rc-rcf)*af**(rf-rfc) + rfv*af**rf)
    return num / (den + 1e-12)
def Func_h2(ac, af, ic, rc, khatc, iff, rf, khatf, iw, rw, khatw, icf, rcf, rfc):
    aw = 1.00001 - ac - af
    rcv, rfv, rwv = ic*khatc/icf, iff*khatf/icf, iw*khatw/icf
    num = 1.0/(iw*khatw)*(rwv*ac**(2-rcf)*af**(2-rfc) - ac*af*aw**(2-rw))
    den = (ac+af)**2 + rfv*ac**(2-rcf)*af**(rf-rfc) + rcv*ac**(rc-rcf)*af**(2-rfc) + aw**(2-rw)/rwv*(rcv*ac**rc + rcv*rfv*ac**(rc-rcf)*af**(rf-rfc) + rfv*af**rf)
    return num / (den + 1e-12)
def func_h2_c_mod(ac, af, ic, rc, khatc, iff, rf, khatf, iw, rw, khatw, icf, rcf, rfc):
    aw = 1.00001 - ac - af
    rcv, rfv, rwv = ic*khatc/icf, iff*khatf/icf, iw*khatw/icf
    num = 1.0/(iw*khatw)*(rwv*ac**(1-rcf)*af**(2-rfc) - af*aw**(2-rw))
    den = (ac+af)**2 + rfv*ac**(2-rcf)*af**(rf-rfc) + rcv*ac**(rc-rcf)*af**(2-rfc) + aw**(2-rw)/rwv*(rcv*ac**rc + rcv*rfv*ac**(rc-rcf)*af**(rf-rfc) + rfv*af**rf)
    return num / (den + 1e-12)
def func_h2_f_mod(ac, af, ic, rc, khatc, iff, rf, khatf, iw, rw, khatw, icf, rcf, rfc):
    aw = 1.00001 - ac - af
    rcv, rfv, rwv = ic*khatc/icf, iff*khatf/icf, iw*khatw/icf
    num = 1.0/(iw*khatw)*(rwv*ac**(2-rcf)*af**(1-rfc) - ac*aw**(2-rw))
    den = (ac+af)**2 + rfv*ac**(2-rcf)*af**(rf-rfc) + rcv*ac**(rc-rcf)*af**(2-rfc) + aw**(2-rw)/rwv*(rcv*ac**rc + rcv*rfv*ac**(rc-rcf)*af**(rf-rfc) + rfv*af**rf)
    return num / (den + 1e-12)
def Func_h3(ac, af, ic, rc, khatc, iff, rf, khatf, iw, rw, khatw, icf, rcf, rfc):
    aw = 1.00001 - ac - af
    rcv, rfv, rwv = ic*khatc/icf, iff*khatf/icf, iw*khatw/icf
    num = aw**(2-rw)/(iw*khatw)*(ac*af + rcv*ac**(rc-rcf)*af**(2-rfc) + af**2)
    den = (ac+af)**2 + rfv*ac**(2-rcf)*af**(rf-rfc) + rcv*ac**(rc-rcf)*af**(2-rfc) + aw**(2-rw)/rwv*(rcv*ac**rc + rcv*rfv*ac**(rc-rcf)*af**(rf-rfc) + rfv*af**rf)
    return num / (den + 1e-12)
def func_h3_f_mod(ac, af, ic, rc, khatc, iff, rf, khatf, iw, rw, khatw, icf, rcf, rfc):
    aw = 1.00001 - ac - af
    rcv, rfv, rwv = ic*khatc/icf, iff*khatf/icf, iw*khatw/icf
    num = aw**(2-rw)/(iw*khatw)*(ac + rcv*ac**(rc-rcf)*af**(1-rfc) + af)
    den = (ac+af)**2 + rfv*ac**(2-rcf)*af**(rf-rfc) + rcv*ac**(rc-rcf)*af**(2-rfc) + aw**(2-rw)/rwv*(rcv*ac**rc + rcv*rfv*ac**(rc-rcf)*af**(rf-rfc) + rfv*af**rf)
    return num / (den + 1e-12)
def func_h3_w_mod(ac, af, ic, rc, khatc, iff, rf, khatf, iw, rw, khatw, icf, rcf, rfc):
    aw = 1.00001 - ac - af
    rcv, rfv, rwv = ic*khatc/icf, iff*khatf/icf, iw*khatw/icf
    num = aw**(1-rw)/(iw*khatw)*(ac*af + rcv*ac**(rc-rcf)*af**(2-rfc) + af**2)
    den = (ac+af)**2 + rfv*ac**(2-rcf)*af**(rf-rfc) + rcv*ac**(rc-rcf)*af**(2-rfc) + aw**(2-rw)/rwv*(rcv*ac**rc + rcv*rfv*ac**(rc-rcf)*af**(rf-rfc) + rfv*af**rf)
    return num / (den + 1e-12)

def func_ff_mod(ac, af, ic, rc, khatc, iff, rf, khatf, iw, rw, khatw, icf, rcf, rfc):
    aw = 1.00001 - ac - af
    rcv, rfv, rwv = ic*khatc/icf, iff*khatf/icf, iw*khatw/icf
    num = ac + af + rcv * ac**(rc-rcf) * af**(1-rfc)
    den = (ac+af)**2 + rfv * ac**(2-rcf) * af**(rf-rfc) + rcv * ac**(rc-rcf) * af**(2-rfc) + \
          aw**(2-rw)/rwv * (rcv * ac**rc + rcv * rfv * ac**(rc-rcf) * af**(rf-rfc) + rfv * af**rf)
    return num / (den + 1e-12)

def func_fw_mod(ac, af, ic, rc, khatc, iff, rf, khatf, iw, rw, khatw, icf, rcf, rfc):
    aw = 1.00001 - ac - af
    rcv, rfv, rwv = ic*khatc/icf, iff*khatf/icf, iw*khatw/icf
    num = aw**(1-rw)/rwv * (rcv * ac**rc + rcv * rfv * ac**(rc-rcf) * af**(rf-rfc) + rfv * af**rf)
    den = (ac+af)**2 + rfv * ac**(2-rcf) * af**(rf-rfc) + rcv * ac**(rc-rcf) * af**(2-rfc) + \
          aw**(2-rw)/rwv * (rcv * ac**rc + rcv * rfv * ac**(rc-rcf) * af**(rf-rfc) + rfv * af**rf)
    return num / (den + 1e-12)
