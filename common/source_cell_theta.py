import numpy as np

def source_cell_theta(alpC, ecm, lam_11, Thetagrow):
    """
    Translation of source_cell_theta.m
    Computes the source term for tumor cell growth.
    """
    # If MATLAB uses the line below, uncomment it. If it uses the other line, comment this one out.
    # Sc = lam_11 * (Thetagrow - alpC)
    
    # Actual Yankeelov version:
    Sc = lam_11 - lam_11 * alpC * Thetagrow
    return Sc