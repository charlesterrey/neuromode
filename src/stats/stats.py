"""
Tests statistiques pour l'analyse S-6.

ANOVA, tests de permutation, corrélations.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional


def anova_two_way(df: pd.DataFrame, dv: str, factor1: str, factor2: str) -> Tuple[float, float, float, float, float, float]:
    """
    ANOVA bidirectionnelle simple via scipy.
    
    Args:
        df: DataFrame des données
        dv: Variable dépendante
        factor1: Premier facteur
        factor2: Second facteur
        
    Returns:
        (F_factor1, p_factor1, F_factor2, p_factor2, F_interaction, p_interaction)
    """
    try:
        from scipy.stats import f_oneway
        
        # Groupes pour factor1
        groups_f1 = [group[dv].values for name, group in df.groupby(factor1)]
        f_stat_f1, p_val_f1 = f_oneway(*groups_f1) if len(groups_f1) > 1 else (0.0, 1.0)
        
        # Groupes pour factor2
        groups_f2 = [group[dv].values for name, group in df.groupby(factor2)]
        f_stat_f2, p_val_f2 = f_oneway(*groups_f2) if len(groups_f2) > 1 else (0.0, 1.0)
        
        # Interaction (approximation simple)
        groups_inter = [group[dv].values for name, group in df.groupby([factor1, factor2])]
        f_stat_inter, p_val_inter = f_oneway(*groups_inter) if len(groups_inter) > 2 else (0.0, 1.0)
        
        return f_stat_f1, p_val_f1, f_stat_f2, p_val_f2, f_stat_inter, p_val_inter
        
    except Exception:
        return 0.0, 1.0, 0.0, 1.0, 0.0, 1.0


def perm_test_diff(df: pd.DataFrame, group_col: str, dv: str, 
                   n_perm: int = 10000, rng: Optional[np.random.Generator] = None) -> float:
    """
    Test de permutation pour différence entre groupes.
    
    Args:
        df: DataFrame des données
        group_col: Colonne des groupes
        dv: Variable dépendante
        n_perm: Nombre de permutations
        rng: Générateur aléatoire
        
    Returns:
        p-value du test de permutation
    """
    if rng is None:
        rng = np.random.default_rng()
    
    groups = df[group_col].unique()
    if len(groups) != 2:
        return 1.0
    
    # Données des deux groupes
    group1_data = df[df[group_col] == groups[0]][dv].values
    group2_data = df[df[group_col] == groups[1]][dv].values
    
    if len(group1_data) == 0 or len(group2_data) == 0:
        return 1.0
    
    # Différence observée
    obs_diff = np.mean(group1_data) - np.mean(group2_data)
    
    # Permutations
    all_data = np.concatenate([group1_data, group2_data])
    n1 = len(group1_data)
    
    perm_diffs = []
    for _ in range(n_perm):
        # Permutation aléatoire
        perm_data = rng.permutation(all_data)
        perm_group1 = perm_data[:n1]
        perm_group2 = perm_data[n1:]
        
        perm_diff = np.mean(perm_group1) - np.mean(perm_group2)
        perm_diffs.append(perm_diff)
    
    # p-value (test bilatéral)
    perm_diffs = np.array(perm_diffs)
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
    
    return float(p_value)


def corr_spearman(df: pd.DataFrame, x: str, y: str) -> Tuple[float, float]:
    """
    Corrélation de Spearman entre deux variables.
    
    Args:
        df: DataFrame des données
        x: Variable x
        y: Variable y
        
    Returns:
        (rho, p_value)
    """
    try:
        # Nettoyage des données
        clean_data = df[[x, y]].dropna()
        
        if len(clean_data) < 3:
            return 0.0, 1.0
        
        # Corrélation de Spearman
        rho, p_value = stats.spearmanr(clean_data[x], clean_data[y])
        
        return float(rho), float(p_value)
        
    except Exception:
        return 0.0, 1.0


def effect_size_eta_squared(f_stat: float, df_effect: int, df_error: int) -> float:
    """
    Calcule eta-carré (taille d'effet) depuis une statistique F.
    
    Args:
        f_stat: Statistique F
        df_effect: Degrés de liberté de l'effet
        df_error: Degrés de liberté de l'erreur
        
    Returns:
        Eta-carré
    """
    if f_stat <= 0:
        return 0.0
    
    eta_sq = (df_effect * f_stat) / (df_effect * f_stat + df_error)
    return float(eta_sq)


def bonferroni_correction(p_values: list, alpha: float = 0.05) -> list:
    """
    Correction de Bonferroni pour tests multiples.
    
    Args:
        p_values: Liste des p-values
        alpha: Seuil de significativité
        
    Returns:
        Liste des p-values corrigées
    """
    n_tests = len(p_values)
    if n_tests == 0:
        return []
    
    corrected = [min(p * n_tests, 1.0) for p in p_values]
    return corrected 