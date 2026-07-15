# Tumor Growth Simulation & Ensemble Kalman Filter (EnKF) Data Assimilation

This repository simulates tumor growth using a three-phase compartment model and
performs data assimilation with an Ensemble Kalman Filter (EnKF) to estimate
model parameters (e.g. `theta0`, `K0`) from noisy observations.

## Project structure

```
tumors_code/
├── common/                # Shared simulation core, physics, and plotting — used by every EnKF method
│   ├── paths.py                        # central path definitions (see "Why this structure" below)
│   ├── A_three_phase_simulator_compartment_full_May08_growth_opt.py   # PDE/compartment simulator
│   ├── solve_Pressure_sparse.py
│   ├── lambda_chem.py
│   ├── lambda_TGF.py
│   ├── source_cell_theta.py
│   ├── source_fibroblast_new.py
│   ├── fastGaussian.py                 # ensemble generation (Gaussian random fields)
│   ├── Plot_EnsemblePred_Iteration_0_HalfFull.py
│   ├── AB_PlotPaper_True_Predict_simulation_ensemb_HalfFull_1_upd.py  # paper-style comparison plots
│   └── Plot_True_simulation_Verify.py
│
├── data/                  # Shared input data (ground truth / initial ensembles), read-only
│   ├── Ensemble_Initial_E20_True_April14_theta0_K0.mat
│   ├── Ensemble_Solution_E20_True_Growth_May08_vary_theta0_K0_reduced.mat
│   ├── new_initial_ensemble_Python_1.mat
│   └── TrueData_May08_theta0_K0_C7.mat
│
├── enkf_sqrt/              # EnKF variant 1: square-root filter
│   ├── sqrtFilter.py                          # the filter/update rule itself
│   ├── runEnsembleOfSimulationsAddNoise.py    # ▶ MAIN SCRIPT — run this
│   └── outputs/            # generated .mat results land here (git-ignored)
│
├── enkf_rlm_mac/        # EnKF variant 2: template for your new method
│   ├── rlm_mac.py                           # ← the rlm-mac
│   ├── rlm_runEnsemble.py  # ▶ run this
│   └── outputs/            # generated .mat results land here (git-ignored)
│
└── README.md
```

## How to run

From the repository root (or from anywhere — paths are resolved automatically, see below):

```bash
pip install -r requirements.txt

python enkf_sqrt/runEnsembleOfSimulationsAddNoise.py
```

This runs the ensemble simulation with noise, applies the square-root EnKF
update, and writes result files to `enkf_sqrt/outputs/`. It also generates
plots via the shared `Plot_*` modules in `common/`.


To run the ensemble simulation with noise, applies the rlm_mac update, and writes results files to enkf_rlm_mac/outputs:

```bash
python enkf_rlm_mac/rlm_runEnsemble.py
```
