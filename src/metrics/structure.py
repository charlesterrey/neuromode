"""
Métriques structurelles pour l'analyse S-6.

Inclut PDI (Plasticity Dependency Index) et métriques de graphe.
"""

import numpy as np
import networkx as nx
from typing import Dict, Any, Optional


def adjacency_from_alive(alive_matrix: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Construit une matrice d'adjacence depuis le masque 'alive'.
    
    Args:
        alive_matrix: Matrice binaire des synapses actives
        weights: Poids synaptiques optionnels
        
    Returns:
        Matrice d'adjacence (binaire ou pondérée)
    """
    if weights is not None:
        # Matrice pondérée
        return alive_matrix.astype(float) * weights
    else:
        # Matrice binaire
        return alive_matrix.astype(float)


def graph_distance_L1(A_ref: np.ndarray, A_cond: np.ndarray) -> float:
    """
    Distance L1 entre deux matrices d'adjacence.
    
    Args:
        A_ref: Matrice de référence
        A_cond: Matrice de condition
        
    Returns:
        Distance L1 = Σ |A_ref - A_cond|
    """
    if A_ref.shape != A_cond.shape:
        raise ValueError("Matrices doivent avoir la même forme")
    
    return float(np.sum(np.abs(A_ref - A_cond)))


def compute_PDI(A_ref: np.ndarray, A_cond: np.ndarray, 
                E_ref: float, E_cond: float, eps: float = 1e-6) -> float:
    """
    Calcule le Plasticity Dependency Index (PDI).
    
    PDI = distance_L1(A_ref, A_cond) / max(|E_ref - E_cond|, ε)
    
    Mesure le changement structural normalisé par le changement énergétique.
    
    Args:
        A_ref: Matrice d'adjacence de référence
        A_cond: Matrice d'adjacence de condition
        E_ref: Énergie de référence
        E_cond: Énergie de condition
        eps: Seuil pour éviter division par zéro
        
    Returns:
        Valeur PDI
    """
    # Distance structurelle
    struct_distance = graph_distance_L1(A_ref, A_cond)
    
    # Différence énergétique
    energy_diff = abs(E_ref - E_cond)
    
    # Normalisation
    if energy_diff < eps:
        return 0.0  # Pas de changement énergétique significatif
    
    pdi = struct_distance / energy_diff
    
    return float(pdi)


def degree_stats(A: np.ndarray) -> Dict[str, float]:
    """
    Calcule les statistiques de degré d'une matrice d'adjacence.
    
    Args:
        A: Matrice d'adjacence
        
    Returns:
        Dictionnaire des statistiques
    """
    # Degrés entrants et sortants
    in_degrees = np.sum(A, axis=0)  # Somme par colonne
    out_degrees = np.sum(A, axis=1)  # Somme par ligne
    total_degrees = in_degrees + out_degrees
    
    stats = {
        'deg_mean': float(np.mean(total_degrees)),
        'deg_std': float(np.std(total_degrees)),
        'deg_max': float(np.max(total_degrees)),
        'deg_min': float(np.min(total_degrees)),
        'in_deg_mean': float(np.mean(in_degrees)),
        'out_deg_mean': float(np.mean(out_degrees)),
        'density': float(np.sum(A > 0) / (A.shape[0] * A.shape[1]))
    }
    
    return stats


def compute_clustering_coefficient(A: np.ndarray) -> float:
    """
    Calcule le coefficient de clustering moyen du graphe.
    
    Args:
        A: Matrice d'adjacence binaire
        
    Returns:
        Coefficient de clustering
    """
    try:
        # Conversion en graphe NetworkX
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        
        # Clustering coefficient moyen
        clustering = nx.average_clustering(G)
        
        return float(clustering)
        
    except Exception:
        return 0.0


def compute_path_lengths(A: np.ndarray, sample_size: int = 100) -> Dict[str, float]:
    """
    Calcule les longueurs de chemins dans le graphe.
    
    Args:
        A: Matrice d'adjacence
        sample_size: Nombre de paires de nœuds à échantillonner
        
    Returns:
        Statistiques des longueurs de chemins
    """
    try:
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        
        # Échantillonnage de paires de nœuds
        nodes = list(G.nodes())
        if len(nodes) < 2:
            return {'avg_path_length': 0.0, 'diameter': 0.0}
        
        # Calcul des chemins les plus courts
        path_lengths = []
        sampled_pairs = min(sample_size, len(nodes) * (len(nodes) - 1))
        
        rng = np.random.default_rng(42)
        for _ in range(sampled_pairs):
            source, target = rng.choice(nodes, 2, replace=False)
            
            try:
                length = nx.shortest_path_length(G, source, target)
                path_lengths.append(length)
            except nx.NetworkXNoPath:
                # Pas de chemin
                continue
        
        if len(path_lengths) == 0:
            return {'avg_path_length': 0.0, 'diameter': 0.0}
        
        stats = {
            'avg_path_length': float(np.mean(path_lengths)),
            'diameter': float(np.max(path_lengths)) if path_lengths else 0.0
        }
        
        return stats
        
    except Exception:
        return {'avg_path_length': 0.0, 'diameter': 0.0}


def network_efficiency(A: np.ndarray, global_eff: bool = True) -> float:
    """
    Calcule l'efficacité du réseau (globale ou locale).
    
    Args:
        A: Matrice d'adjacence
        global_eff: Si True, efficacité globale; sinon locale
        
    Returns:
        Efficacité du réseau
    """
    try:
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        
        if global_eff:
            efficiency = nx.global_efficiency(G)
        else:
            efficiency = nx.local_efficiency(G)
        
        return float(efficiency)
        
    except Exception:
        return 0.0


def structural_summary(A: np.ndarray) -> Dict[str, Any]:
    """
    Résumé complet des propriétés structurelles d'un réseau.
    
    Args:
        A: Matrice d'adjacence
        
    Returns:
        Dictionnaire des métriques structurelles
    """
    summary = {}
    
    # Statistiques de degré
    summary.update(degree_stats(A))
    
    # Clustering
    summary['clustering'] = compute_clustering_coefficient(A)
    
    # Efficacité
    summary['global_efficiency'] = network_efficiency(A, global_eff=True)
    summary['local_efficiency'] = network_efficiency(A, global_eff=False)
    
    # Longueurs de chemins
    path_stats = compute_path_lengths(A)
    summary.update(path_stats)
    
    # Métriques de base
    n_nodes = A.shape[0]
    n_edges = np.sum(A > 0)
    
    summary.update({
        'n_nodes': int(n_nodes),
        'n_edges': int(n_edges),
        'edge_density': float(n_edges / (n_nodes * n_nodes)) if n_nodes > 0 else 0.0
    })
    
    return summary 