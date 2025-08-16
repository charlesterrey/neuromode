"""
Métriques comportementales pour l'analyse S-6.

Inclut ODI et intervalles de confiance bootstrap.
"""

import numpy as np
from typing import Optional, Tuple


def compute_ODI(recall_noassist: float, recall_assist: float) -> float:
    """
    Calcule l'Offloading Dependency Index (ODI).
    
    ODI = rappel_sans_aide - rappel_avec_aide
    
    ODI > 0 : Performance meilleure sans aide (autonomie)
    ODI < 0 : Performance meilleure avec aide (dépendance)
    
    Args:
        recall_noassist: Score de rappel sans assistance
        recall_assist: Score de rappel avec assistance
        
    Returns:
        Valeur ODI
    """
    return float(recall_noassist - recall_assist)


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, 
                 rng: Optional[np.random.Generator] = None) -> Tuple[float, float, float]:
    """
    Calcule l'intervalle de confiance bootstrap pour la moyenne.
    
    Args:
        x: Échantillon de données
        n_boot: Nombre d'échantillons bootstrap
        alpha: Niveau de risque (0.05 pour IC 95%)
        rng: Générateur aléatoire (None pour défaut)
        
    Returns:
        (moyenne, borne_inf, borne_sup)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if len(x) == 0:
        return 0.0, 0.0, 0.0
    
    # Échantillonnage bootstrap
    n = len(x)
    bootstrap_means = []
    
    for _ in range(n_boot):
        # Échantillonnage avec remise
        bootstrap_sample = rng.choice(x, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Calcul des percentiles
    mean_est = np.mean(x)
    ci_low = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return float(mean_est), float(ci_low), float(ci_high)


def effect_size_cohen_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Calcule la taille d'effet de Cohen (d) entre deux groupes.
    
    d = (μ₁ - μ₂) / σ_pooled
    
    Args:
        x1: Échantillon groupe 1
        x2: Échantillon groupe 2
        
    Returns:
        Taille d'effet Cohen's d
    """
    if len(x1) == 0 or len(x2) == 0:
        return 0.0
    
    # Moyennes
    m1, m2 = np.mean(x1), np.mean(x2)
    
    # Écarts-types
    s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    
    # Écart-type poolé
    n1, n2 = len(x1), len(x2)
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    
    if s_pooled == 0:
        return 0.0
    
    # Cohen's d
    d = (m1 - m2) / s_pooled
    
    return float(d)


def summarize_odi_by_condition(df, group_cols=['omega', 't0_ms'], value_col='ODI') -> dict:
    """
    Résume l'ODI par condition avec statistiques descriptives.
    
    Args:
        df: DataFrame des résultats
        group_cols: Colonnes de groupement
        value_col: Colonne des valeurs ODI
        
    Returns:
        Dictionnaire des statistiques par condition
    """
    summary = {}
    
    for name, group in df.groupby(group_cols):
        if len(group) == 0:
            continue
        
        values = group[value_col].dropna().values
        
        if len(values) == 0:
            summary[str(name)] = {
                'n': 0, 'mean': 0.0, 'std': 0.0, 
                'ci_low': 0.0, 'ci_high': 0.0
            }
            continue
        
        # Statistiques descriptives
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
        
        # Intervalle de confiance bootstrap
        mean_boot, ci_low, ci_high = bootstrap_ci(values)
        
        summary[str(name)] = {
            'n': len(values),
            'mean': float(mean_val),
            'std': float(std_val),
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    return summary 