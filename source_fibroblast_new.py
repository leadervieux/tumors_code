import numpy as np

def source_fibroblast_new(alpF, ecm, lam_61, lam_62, lam_63):
    """
    Traduction de source_fibroblast_new.m
    Calcule le terme source pour la croissance des fibroblastes.
    """
    # Sf = lam_61*alpF - lam_62*alpF.^2 - lam_63*ecm.*alpF; (version commentée dans MATLAB)
    Sf = 0.0 # Dans le simulateur A, lam_61, lam_62, lam_63 sont souvent mis à 0, donc Sf est 0.
    return Sf