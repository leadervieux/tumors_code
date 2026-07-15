import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat



THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "common"))
sys.path.insert(0, THIS_DIR)

from paths import DATA_DIR, output_dir  # noqa: E402

# --- Configuration RLM-MAC (iES) ---
pdim = 61 * 61

parser = argparse.ArgumentParser(description="Compare les ensembles RLM-MAC (iES) à la vérité terrain.")
parser.add_argument(
    "--dir", dest="dir_path", default=None,
    help="Dossier contenant les ensemble*.mat / objRealIter*.mat "
         "(par défaut : enkf_rlm_mac/outputs, à côté de ce script).",
)
args, _ = parser.parse_known_args()

dir_path = args.dir_path if args.dir_path is not None else output_dir(THIS_DIR)
init_file = os.path.join(dir_path, 'initialEnsemble_Python_rlm.mat')
true_file = os.path.join(DATA_DIR, 'TrueData_May08_theta0_K0_C7.mat')
full_shape = (61, 61)
mask_center = (30, 30)

radius_fibro = 20
radius_kw = 16
radius_tv = 9
y_max = 50


def build_circular_mask(shape, center=None, radius=None):
    """Return a boolean mask selecting pixels inside a disk."""
    if center is None:
        center = (shape[1] // 2, shape[0] // 2)
    if radius is None:
        radius = min(shape) // 2 - 1

    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return ((xx - center[0]) ** 2 + (yy - center[1]) ** 2) <= radius**2


def build_variable_masks(fibro_radius, kw_radius, tv_radius, y_max):
    mask_fibro = build_circular_mask(full_shape, center=mask_center, radius=fibro_radius)
    mask_kw = build_circular_mask(full_shape, center=mask_center, radius=kw_radius)
    mask_tv = build_circular_mask(full_shape, center=mask_center, radius=tv_radius)

    if y_max is not None:
        mask_fibro &= np.arange(full_shape[0])[:, None] <= y_max
        mask_kw &= np.arange(full_shape[0])[:, None] <= y_max
        mask_tv &= np.arange(full_shape[0])[:, None] <= y_max

    return mask_fibro, mask_kw, mask_tv

mask_fibro, mask_kw, mask_tv = build_variable_masks(radius_fibro, radius_kw, radius_tv, y_max)


# --- Loading initial ensemble ---
if not os.path.exists(init_file):
    raise FileNotFoundError(f"The files for the iteration 0 is missing : {init_file}")

initial_data = loadmat(init_file)
# initialEnsemble_Python_rlm.mat utilise la clé 'initialEnsemble' (pas 'ensemble')
if 'initialEnsemble' not in initial_data:
    raise KeyError(f"The key 'initialEnsemble' is missing from {init_file}")

true_data = loadmat(true_file)
if 'initialTrueModel' not in true_data:
    raise KeyError(f"initialTrueModel is missing from {true_file}")

initial_ensemble_raw = np.asarray(initial_data['initialEnsemble'])
if initial_ensemble_raw.shape[0] != 6 * pdim and initial_ensemble_raw.shape[1] == 6 * pdim:
    initial_ensemble_raw = initial_ensemble_raw.T
if initial_ensemble_raw.shape[0] != 6 * pdim:
    raise ValueError(f"initialEnsemble invalide, shape={initial_ensemble_raw.shape}")

# Modifié : Retrait de la dernière colonne (la moyenne), si présente dans ce fichier.
# À ajuster si initialEnsemble_Python_rlm.mat ne contient pas de colonne de moyenne en plus des membres.
initial_ensemble = initial_ensemble_raw[:, :]

true_ensemble = np.asarray(true_data['initialTrueModel'])
if true_ensemble.shape[0] != 6 * pdim and true_ensemble.shape[1] == 6 * pdim:
    true_ensemble = true_ensemble.T
if true_ensemble.shape[0] != 6 * pdim:
    raise ValueError(f"initialEnsemble_True invalide, shape={true_ensemble.shape}")

# Référence “vraie” : premier membre
truth_reference = true_ensemble[:, 0]


# Visualisation des masques au début
variable_names = ['fibroblastes', 'log(k_w)', 'T_v']
variable_slices = [
    (0, pdim),
    (pdim, 2 * pdim),
    (2 * pdim, 3 * pdim),
]

for var_name, (start, stop) in zip(variable_names, variable_slices):
    block = initial_ensemble[start:stop, 0].reshape(61, 61)
    block = np.asarray(block, dtype=float)
    block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
    mask_for_var = mask_fibro if var_name == 'fibroblastes' else mask_kw if var_name == 'log(k_w)' else mask_tv
    plt.figure(figsize=(5, 4))
    plt.imshow(block, origin='lower', cmap='viridis')
    plt.contour(mask_for_var, levels=[0.5], colors='white', linewidths=1.0)
    radius = radius_fibro if var_name == 'fibroblastes' else radius_kw if var_name == 'log(k_w)' else mask_tv
    plt.title(f'{var_name} - masque (radius={radius})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.show()


# --- Traitement des fichiers d'itérations ---
# Trie numériquement les fichiers (pour éviter par exemple que ensemble10 passe avant ensemble2)
def extract_number(filename):
    s = re.findall(r'\d+', os.path.basename(filename))
    return int(s[0]) if s else 0

# On cherche les fichiers ensemble*.mat du dossier outputs (ensemble1, ensemble2, ...)
# L'ensemble initial est chargé séparément depuis initialEnsemble_Python_rlm.mat (voir plus haut),
# donc pas besoin de l'exclure ici.
iteration_files = sorted(
    glob.glob(os.path.join(dir_path, "ensemble[0-9]*.mat")),
    key=extract_number
)

if not iteration_files:
    raise FileNotFoundError(f"No iteration files found for iES in {dir_path}")


errors_fibro = []
errors_kw = []
errors_Tv = []


def error_against_reference(state, reference):
    return np.linalg.norm(state - reference[:, None], axis=0)


def restrict_to_mask(block, mask_1d):
    arr = np.asarray(block)
    mask = np.asarray(mask_1d, dtype=bool).ravel()
    if mask.size != arr.shape[0]:
        raise ValueError(f"Mask size {mask.size} does not match array length {arr.shape[0]}")
    if arr.ndim == 1:
        return arr[mask]
    return arr[mask, :]


# --- Itération 0 : ensemble initial ---
init_fibro = restrict_to_mask(initial_ensemble[0:pdim, :-1], mask_fibro)
ref_fibro = restrict_to_mask(truth_reference[0:pdim], mask_fibro)
errors_fibro.append(error_against_reference(init_fibro, ref_fibro))

# Modifié : Pas de np.log sur initial_ensemble[pdim:2*pdim] car khat_w est déjà stocké en log
kw_init = restrict_to_mask(initial_ensemble[pdim:2 * pdim, :-1], mask_kw)
kw_true = np.log(np.clip(restrict_to_mask(truth_reference[pdim:2* pdim], mask_kw), 1e-12, None)) # reste converti car la verité est en espace direct
errors_kw.append(error_against_reference(kw_init, kw_true))

Tv_init = restrict_to_mask(initial_ensemble[2 * pdim:3 * pdim, :-1], mask_tv)
Tv_true = restrict_to_mask(truth_reference[2 * pdim:3 * pdim], mask_tv)
errors_Tv.append(error_against_reference(Tv_init, Tv_true))

labels = ['iter0']


# --- Boucle sur les itérations estimées ---
for iteration_file in iteration_files:
    data = loadmat(iteration_file)
    # Modifié : Clé RLM-MAC 'ensemble'
    if 'ensemble' not in data:
        raise KeyError(f"The key 'ensemble' is missing from {iteration_file}")
    
    updated_raw = np.asarray(data['ensemble'])
    num_iter = extract_number(iteration_file)
    label = f'iter{num_iter}'

    if updated_raw.shape[0] != 6 * pdim and updated_raw.shape[1] == 6 * pdim:
        updated_raw = updated_raw.T
    if updated_raw.shape[0] != 6 * pdim:
        raise ValueError(f"Invalid ensemble in {iteration_file}, shape={updated_raw.shape}")
    
    # Modifié : Retrait de la dernière colonne (la moyenne)
    updated = updated_raw[:, :-1]

    if updated.shape[1] != initial_ensemble.shape[1]:
        raise ValueError(f"Invalid ensemble size between {init_file} and {iteration_file}: {initial_ensemble.shape[1]} vs {updated.shape[1]}")

    fibros_upd = restrict_to_mask(updated[0:pdim, :-1], mask_fibro)
    err_fibro = error_against_reference(fibros_upd, restrict_to_mask(truth_reference[0:pdim], mask_fibro))

    # Modifié : Pas de np.log car khat_w est déjà stocké en log dans l'ensemble iES
    kw_upd = (restrict_to_mask(updated[pdim:2 * pdim, :-1], mask_kw))
    err_kw = error_against_reference(kw_upd, kw_true)

    Tv_upd = restrict_to_mask(updated[2 * pdim:3 * pdim, :-1], mask_tv)
    err_Tv = error_against_reference(Tv_upd, restrict_to_mask(truth_reference[2 * pdim:3 * pdim], mask_tv))

    errors_fibro.append(err_fibro)
    errors_kw.append(err_kw)
    errors_Tv.append(err_Tv)
    labels.append(label)

    print(f"Traité {iteration_file} : {updated.shape[1]} membres ({label})")


# --- Tracé des boxplots ---
plt.figure(figsize=(10, 4))
plt.boxplot(errors_fibro, tick_labels=labels, showmeans=True)
plt.title('RMS fibroblastes each iteration (iES)')
plt.xlabel('Itération')
plt.ylabel('RMS error')
plt.grid(True, linestyle=':', alpha=0.5)

plt.figure(figsize=(10, 4))
plt.boxplot(errors_kw, tick_labels=labels, showmeans=True)
plt.title('RMS log(k_w) each iteration (iES)')
plt.xlabel('Itération')
plt.ylabel('RMS error')
plt.grid(True, linestyle=':', alpha=0.5)

plt.figure(figsize=(10, 4))
plt.boxplot(errors_Tv, tick_labels=labels, showmeans=True)
plt.title('RMS T_v each iteration (iES)')
plt.xlabel('Itération')
plt.ylabel('RMS error')
plt.grid(True, linestyle=':', alpha=0.5)



obj_files = sorted(
    glob.glob(os.path.join(dir_path, "objRealIter*.mat")),
    key=extract_number
)

if not obj_files:
    print("No objRealIter*.mat files found.")
else:

    obj_all_iters = []
    obj_labels = []
    obj_means = []

    for f in obj_files:
        num_iter = extract_number(f)
        data = loadmat(f)
        
        # 'objReal' contains 100 values
        obj_values = np.squeeze(data['objReal']) 
        
        obj_all_iters.append(obj_values)
        obj_labels.append(f'iter{num_iter}')
        obj_means.append(np.mean(obj_values)) # On calcule la moyenne de l'ensemble

    # --- 3. Boxplot with distribution of the 100 members after each iteration ---
    plt.figure(15, figsize=(10, 4))
    plt.boxplot(obj_all_iters, tick_labels=obj_labels, showmeans=True)
    plt.xlabel('Iteration')
    plt.ylabel('Objective function (Misfit)')
    plt.title('Distribution of the objective function per iteration (iES)')
    plt.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()


plt.tight_layout()
plt.show()