import numpy as np

def source_fibroblast_new(alpF, ecm, lam_61, lam_62, lam_63):
    """
    Translation of source_fibroblast_new.m
    Computes the source term for fibroblast growth.
    """
    # Sf = lam_61*alpF - lam_62*alpF.^2 - lam_63*ecm.*alpF; (commented version in MATLAB)
    Sf = 0.0 # In the simulator A, lam_61, lam_62, lam_63 are often set to 0, so Sf is 0.
    return Sf