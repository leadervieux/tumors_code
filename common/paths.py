"""
Central path definitions.

Every run script imports from here instead of using bare filenames like
'Ensemble_Initial_E20_True_April14_theta0_K0.mat'. This is what lets the
project be organised into folders (common/, data/, enkf_sqrt/, ...)
without breaking file loading: paths are computed from this file's own
location, not from the directory the user happens to launch Python from.
"""
import os

COMMON_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(COMMON_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")


def output_dir(method_dir):
    """
    Return the outputs/ folder that lives next to a given EnKF method
    (e.g. enkf_sqrt/outputs or enkf_new_method/outputs), creating it if
    needed. Each method writes its generated .mat files there instead of
    polluting the repo root or clashing with another method's outputs.
    """
    out = os.path.join(method_dir, "outputs")
    os.makedirs(out, exist_ok=True)
    return out
