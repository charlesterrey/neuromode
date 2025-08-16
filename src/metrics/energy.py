"""
Métriques énergétiques pour l'analyse S-6.

Agrégation des données énergétiques des simulations S-4.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def energy_aggregates(energy_csv_path: str) -> Dict[str, float]:
    """
    Calcule les agrégats énergétiques depuis un fichier CSV S-4.
    
    Args:
        energy_csv_path: Chemin vers le fichier energy.csv
        
    Returns:
        Dictionnaire des métriques énergétiques
    """
    try:
        # Chargement des données
        df = pd.read_csv(energy_csv_path)
        
        # Colonnes attendues: t_ms, E_win, P_E, budget_B
        if not all(col in df.columns for col in ['t_ms', 'E_win', 'P_E']):
            # Fallback si structure différente
            return _fallback_energy_metrics()
        
        # Calculs d'agrégation
        E_win = df['E_win'].values
        P_E = df['P_E'].values
        
        # Énergie totale (somme cumulée)
        E_total = float(np.sum(E_win))
        
        # Nombre de dépassements de budget
        if 'budget_B' in df.columns:
            budget = df['budget_B'].iloc[0]  # Budget constant
            overshoot_count = int(np.sum(E_win > budget))
        else:
            overshoot_count = int(np.sum(P_E > 0))  # Utiliser P_E comme proxy
        
        # Statistiques des fenêtres énergétiques
        E_mean_win = float(np.mean(E_win))
        E_std_win = float(np.std(E_win))
        E_max_win = float(np.max(E_win))
        E_min_win = float(np.min(E_win))
        
        # Pression énergétique
        P_E_mean = float(np.mean(P_E))
        P_E_max = float(np.max(P_E))
        
        # Évolution temporelle
        if len(E_win) > 1:
            E_trend = float(np.polyfit(range(len(E_win)), E_win, 1)[0])  # Pente
        else:
            E_trend = 0.0
        
        aggregates = {
            'E_total': E_total,
            'E_overshoot': float(overshoot_count),
            'E_mean_win': E_mean_win,
            'E_std_win': E_std_win,
            'E_max_win': E_max_win,
            'E_min_win': E_min_win,
            'P_E_mean': P_E_mean,
            'P_E_max': P_E_max,
            'E_trend': E_trend,
            'n_windows': len(E_win)
        }
        
        return aggregates
        
    except Exception as e:
        # En cas d'erreur, retourner des valeurs par défaut
        return _fallback_energy_metrics()


def _fallback_energy_metrics() -> Dict[str, float]:
    """
    Métriques énergétiques par défaut en cas d'erreur.
    
    Returns:
        Dictionnaire avec valeurs par défaut
    """
    return {
        'E_total': 0.0,
        'E_overshoot': 0.0,
        'E_mean_win': 0.0,
        'E_std_win': 0.0,
        'E_max_win': 0.0,
        'E_min_win': 0.0,
        'P_E_mean': 0.0,
        'P_E_max': 0.0,
        'E_trend': 0.0,
        'n_windows': 0
    }


def compare_energy_profiles(energy_csv_ref: str, energy_csv_cond: str) -> Dict[str, float]:
    """
    Compare deux profils énergétiques.
    
    Args:
        energy_csv_ref: Fichier énergie de référence
        energy_csv_cond: Fichier énergie de condition
        
    Returns:
        Métriques de comparaison
    """
    try:
        ref_agg = energy_aggregates(energy_csv_ref)
        cond_agg = energy_aggregates(energy_csv_cond)
        
        # Différences relatives
        comparison = {}
        
        for key in ['E_total', 'E_mean_win', 'E_max_win', 'P_E_mean']:
            ref_val = ref_agg.get(key, 0.0)
            cond_val = cond_agg.get(key, 0.0)
            
            if ref_val != 0:
                rel_diff = (cond_val - ref_val) / ref_val
            else:
                rel_diff = 0.0
            
            comparison[f'delta_{key}'] = float(rel_diff)
        
        # Différences absolues
        comparison['delta_E_overshoot'] = float(cond_agg['E_overshoot'] - ref_agg['E_overshoot'])
        
        return comparison
        
    except Exception:
        return {'delta_E_total': 0.0, 'delta_E_mean_win': 0.0, 
                'delta_E_max_win': 0.0, 'delta_P_E_mean': 0.0,
                'delta_E_overshoot': 0.0}


def energy_efficiency_score(E_total: float, performance_metric: float, 
                           baseline_E: float = 50000.0) -> float:
    """
    Calcule un score d'efficacité énergétique.
    
    Efficacité = performance / (E_total / baseline_E)
    
    Args:
        E_total: Énergie totale consommée
        performance_metric: Métrique de performance (ex: ODI, rappel)
        baseline_E: Énergie de référence
        
    Returns:
        Score d'efficacité énergétique
    """
    if E_total <= 0:
        return 0.0
    
    # Normalisation énergétique
    energy_ratio = E_total / baseline_E
    
    # Score d'efficacité
    efficiency = performance_metric / energy_ratio
    
    return float(efficiency)


def summarize_energy_by_condition(df, group_cols=['omega', 't0_ms']) -> Dict[str, Dict[str, float]]:
    """
    Résume les métriques énergétiques par condition.
    
    Args:
        df: DataFrame des résultats
        group_cols: Colonnes de groupement
        
    Returns:
        Dictionnaire des statistiques énergétiques par condition
    """
    summary = {}
    
    energy_cols = ['E_total', 'E_overshoot', 'E_mean_win', 'P_E_mean']
    
    for name, group in df.groupby(group_cols):
        if len(group) == 0:
            continue
        
        cond_summary = {}
        
        for col in energy_cols:
            if col in group.columns:
                values = group[col].dropna()
                if len(values) > 0:
                    cond_summary[f'{col}_mean'] = float(values.mean())
                    cond_summary[f'{col}_std'] = float(values.std()) if len(values) > 1 else 0.0
                else:
                    cond_summary[f'{col}_mean'] = 0.0
                    cond_summary[f'{col}_std'] = 0.0
        
        summary[str(name)] = cond_summary
    
    return summary 