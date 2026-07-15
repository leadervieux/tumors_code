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
├── enkf_new_method/        # EnKF variant 2: template for your new method
│   ├── rlm_mac.py                           # ← replace with your new update rule
│   ├── rlm_runEnsemble.py  # ▶ run this once newFilter.py is filled in
│   └── outputs/            # generated .mat results land here (git-ignored)
│
└── README.md
```

## How to run

From the repository root (or from anywhere — paths are resolved automatically, see below):

```bash
pip install numpy scipy matplotlib scikit-image

python enkf_sqrt/runEnsembleOfSimulationsAddNoise.py
```

This runs the ensemble simulation with noise, applies the square-root EnKF
update, and writes result files to `enkf_sqrt/outputs/`. It also generates
plots via the shared `Plot_*` modules in `common/`.

To compare against the reference/paper figures:

```bash
python -c "from enkf_sqrt.AB_PlotPaper_True_Predict_simulation_ensemb_HalfFull_1_upd import plot_paper_comparison; plot_paper_comparison(K_member=6)"
```

Once you've written your new filter in `enkf_new_method/newFilter.py`:

```bash
python enkf_new_method/runEnsembleOfSimulations_NewMethod.py
```

## Why this structure? (folders without breaking imports)

The original code was a flat list of files: every script imported the others
by bare filename (`from sqrtFilter import sqrt_filter`) and loaded data by
bare filename (`loadmat('Ensemble_Initial_...mat')`). Both of these only work
if every file lives in the same directory *and* you run the script from that
exact directory — which is exactly what breaks the moment you organize things
into folders.

Two small fixes solve this permanently:

1. **`common/paths.py`** computes `DATA_DIR` (and an `output_dir()` helper)
   from its own file location using `os.path.abspath(__file__)`, not from the
   current working directory. So `DATA_DIR` always points at `data/` no
   matter where you launch Python from.

2. Each **run script** (`runEnsembleOfSimulationsAddNoise.py`,
   `runEnsembleOfSimulations_NewMethod.py`) starts with a small bootstrap
   block that adds `common/` to `sys.path` before importing anything from it:

   ```python
   THIS_DIR = os.path.dirname(os.path.abspath(__file__))
   REPO_ROOT = os.path.dirname(THIS_DIR)
   sys.path.insert(0, os.path.join(REPO_ROOT, "common"))
   sys.path.insert(0, THIS_DIR)

   from paths import DATA_DIR, output_dir
   OUTPUT_DIR = output_dir(THIS_DIR)
   ```

   After that, `from fastGaussian import fast_gaussian`,
   `from sqrtFilter import sqrt_filter`, etc. all resolve exactly as they did
   in the original flat layout — you don't need to touch any other import in
   the codebase.

All `loadmat(...)` / `savemat(...)` calls that used to reference bare
filenames now use `DATA_DIR` (for shared truth/initial-condition inputs) or
`OUTPUT_DIR` (for each method's generated results), so two EnKF methods can
run side by side without overwriting each other's output files.

## Adding a third EnKF method later

Copy `enkf_new_method/` as a template:

```bash
cp -r enkf_new_method enkf_<your_method_name>
```

Rename the filter file and update the one import line at the top of the run
script to point at it. Everything else (simulator, plotting, data) is shared
automatically through `common/` and `data/` — no other changes needed.

## Notes

- `new_initial_ensemble_Python_1.mat` (loaded by the main run script) is not
  tracked in this repo. If you regenerate it elsewhere, drop it in `data/` so
  it resolves through `DATA_DIR`.
- The filenames read by `AB_PlotPaper_..._upd.py` for `enkf_res` /
  `enkf_pred` should match whatever your run script actually saved in
  `enkf_sqrt/outputs/` — rename either side if they drift apart.
