import numpy as np

def source_cell_theta(alpC, ecm, lam_11, Thetagrow):
    """
    Traduction de source_cell_theta.m
    Calcule le terme source pour la croissance des cellules tumorales.
    """
    # ATTENTION : Vérifie quelle ligne est décommentée dans ton MATLAB
    # Si MATLAB utilise la version linéaire, décommente la ligne suivante :
    # Sc = lam_11 * (Thetagrow - alpC)
    
    # Version Yankeelov actuelle :
    Sc = lam_11 - lam_11 * alpC * Thetagrow
    return Sc