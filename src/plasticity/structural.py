"""
Utilitaires pour la plasticité structurelle développementale.

Implémente les mécanismes de surcroissance et d'élagage synaptique
basés sur l'activité et les coûts de câblage.
"""

import numpy as np
import networkx as nx
from typing import Tuple, Dict, Any, Optional
from ..utils.io import log_info, log_warning


def sigmoid(x: np.ndarray, k: float = 1.0) -> np.ndarray:
    """Fonction sigmoïde pour probabilités de pruning."""
    return 1.0 / (1.0 + np.exp(-k * x))


def compute_density(alive_vector: np.ndarray, N: int) -> float:
    """
    Calcule la densité synaptique.
    
    Args:
        alive_vector: Vecteur des connexions actives (0/1)
        N: Nombre de neurones dans la population
    
    Returns:
        Densité synaptique (nb_connexions / nb_connexions_possibles)
    """
    nb_alive = np.sum(alive_vector)
    nb_possible = N * (N - 1)  # Connexions possibles (sans auto-connexions)
    return nb_alive / nb_possible if nb_possible > 0 else 0.0


def sampling_indices_for_monitor(n: int, max_n: int) -> np.ndarray:
    """
    Génère des indices d'échantillonnage pour le monitoring.
    
    Args:
        n: Nombre total d'éléments
        max_n: Nombre maximum à échantillonner
    
    Returns:
        Indices échantillonnés
    """
    if n <= max_n:
        return np.arange(n)
    else:
        return np.random.choice(n, size=max_n, replace=False)


def pdi_graph_edit(adj_prev: np.ndarray, adj_curr: np.ndarray, 
                  threshold: float = 0.5) -> float:
    """
    Calcule une approximation rapide du PDI (Proportion of Developmental Instability)
    basée sur la distance d'édition de graphes.
    
    Args:
        adj_prev: Matrice d'adjacence précédente (binaire ou pondérée)
        adj_curr: Matrice d'adjacence actuelle
        threshold: Seuil pour binariser les matrices pondérées
    
    Returns:
        Score PDI approximatif (0-1, 1 = changement maximal)
    """
    # Binarisation des matrices
    prev_bin = (adj_prev > threshold).astype(int)
    curr_bin = (adj_curr > threshold).astype(int)
    
    # Calcul des changements
    edges_added = np.sum((prev_bin == 0) & (curr_bin == 1))
    edges_removed = np.sum((prev_bin == 1) & (curr_bin == 0))
    total_changes = edges_added + edges_removed
    
    # Normalisation par le nombre total d'arêtes possibles
    n = adj_prev.shape[0]
    max_edges = n * (n - 1)  # Graphe dirigé sans auto-boucles
    
    if max_edges == 0:
        return 0.0
    
    return total_changes / max_edges


def grow_step(syn_ee: Any, target_density: float, max_add_per_step: int,
             rng: np.random.Generator) -> int:
    """
    Étape de surcroissance : active aléatoirement des synapses inactives.
    
    Args:
        syn_ee: Objet Synapses Brian2 E→E
        target_density: Densité cible à atteindre
        max_add_per_step: Nombre maximum de synapses à activer par étape
        rng: Générateur de nombres aléatoires
    
    Returns:
        Nombre de synapses activées
    """
    # Calcul de la densité actuelle
    N_e = syn_ee.source.N
    current_density = compute_density(syn_ee.alive, N_e)
    
    if current_density >= target_density:
        return 0  # Densité cible atteinte
    
    # Trouver les synapses inactives
    inactive_indices = np.where(syn_ee.alive == 0)[0]
    
    if len(inactive_indices) == 0:
        return 0  # Toutes les synapses sont actives
    
    # Calculer le nombre à activer
    total_synapses = len(syn_ee.alive)
    target_count = int(target_density * total_synapses)
    current_count = int(np.sum(syn_ee.alive))
    needed = min(target_count - current_count, max_add_per_step, len(inactive_indices))
    
    if needed <= 0:
        return 0
    
    # Sélection aléatoire des synapses à activer
    to_activate = rng.choice(inactive_indices, size=needed, replace=False)
    
    # Activation des synapses sélectionnées
    syn_ee.alive[to_activate] = 1.0
    
    # Initialisation des poids pour les nouvelles connexions
    syn_ee.w[to_activate] = rng.normal(0.5, 0.1, size=needed)
    syn_ee.w[to_activate] = np.clip(syn_ee.w[to_activate], 0, 1.0)
    
    # Réinitialisation des traces d'activité
    syn_ee.act_score[to_activate] = 0.0
    
    return needed


def prune_step(syn_ee: Any, theta_act: float, k1: float, k2: float,
              rng: np.random.Generator, reset_A_fraction: float = 0.5) -> int:
    """
    Étape d'élagage : supprime des synapses selon la règle activité + coût.
    
    La probabilité de pruning suit :
    p_prune = 1 / (1 + exp(-(k1*(theta_act - A) + k2*len_cost)))
    
    Args:
        syn_ee: Objet Synapses Brian2 E→E
        theta_act: Seuil d'activité pour l'élagage
        k1: Poids du terme d'activité
        k2: Poids du coût de câblage
        rng: Générateur de nombres aléatoires
        reset_A_fraction: Fraction de A à réinitialiser après élagage
    
    Returns:
        Nombre de synapses élaguées
    """
    # Trouver les synapses actives
    active_indices = np.where(syn_ee.alive == 1)[0]
    
    if len(active_indices) == 0:
        return 0  # Aucune synapse active
    
    # Calcul des probabilités de pruning
    A_active = syn_ee.act_score[active_indices]
    len_cost_active = syn_ee.len_cost[active_indices]
    
    # Score de pruning : activité faible + coût élevé = plus probable d'être élaguée
    prune_score = k1 * (theta_act - A_active) + k2 * len_cost_active
    p_prune = sigmoid(prune_score, k=1.0)
    
    # Tirage aléatoire pour déterminer quelles synapses élaguer
    random_vals = rng.random(len(active_indices))
    to_prune_mask = random_vals < p_prune
    to_prune = active_indices[to_prune_mask]
    
    if len(to_prune) == 0:
        return 0
    
    # Élagage des synapses sélectionnées
    syn_ee.alive[to_prune] = 0.0
    syn_ee.w[to_prune] = 0.0
    
    # Réinitialisation partielle des scores d'activité (survie des plus aptes)
    if reset_A_fraction > 0:
        # Réinitialiser partiellement A pour toutes les synapses survivantes
        surviving_indices = np.where(syn_ee.alive == 1)[0]
        if len(surviving_indices) > 0:
            decay_factor = 1.0 - reset_A_fraction
            syn_ee.act_score[surviving_indices] *= decay_factor
    
    return len(to_prune)


def create_position_grid(N: int, grid_size: float = 10.0) -> np.ndarray:
    """
    Crée des positions 2D pour les neurones sur une grille.
    
    Args:
        N: Nombre de neurones
        grid_size: Taille de la grille
    
    Returns:
        Array de positions (N, 2)
    """
    # Arrangement en grille approximativement carrée
    side_length = int(np.ceil(np.sqrt(N)))
    
    positions = []
    for i in range(N):
        x = (i % side_length) * (grid_size / side_length)
        y = (i // side_length) * (grid_size / side_length)
        positions.append([x, y])
    
    return np.array(positions)


def compute_distance_costs(positions_pre: np.ndarray, positions_post: np.ndarray,
                          lambda_len: float = 1.0) -> np.ndarray:
    """
    Calcule les coûts de câblage normalisés entre neurones pré et post.
    
    Args:
        positions_pre: Positions des neurones présynaptiques (N_pre, 2)
        positions_post: Positions des neurones postsynaptiques (N_post, 2)
        lambda_len: Facteur de normalisation pour les distances
    
    Returns:
        Matrice des coûts (N_pre * (N_post-1),) pour connexions full sans auto-connexions
    """
    costs = []
    
    for i_pre in range(len(positions_pre)):
        for i_post in range(len(positions_post)):
            if i_pre != i_post:  # Pas d'auto-connexions
                distance = np.linalg.norm(positions_pre[i_pre] - positions_post[i_post])
                cost = distance / lambda_len
                costs.append(cost)
            # On n'ajoute rien pour les auto-connexions (i_pre == i_post)
    
    return np.array(costs)


def log_structural_event(time_ms: float, event_type: str, count: int, 
                        density: float, outdir: str) -> None:
    """
    Enregistre un événement structural dans un fichier log.
    
    Args:
        time_ms: Temps de simulation
        event_type: Type d'événement ("GROW" ou "PRUNE")
        count: Nombre de synapses affectées
        density: Densité synaptique actuelle
        outdir: Répertoire de sortie
    """
    import os
    log_file = os.path.join(outdir, "structural_events.csv")
    
    # Créer le fichier avec en-têtes s'il n'existe pas
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("time_ms,event_type,count,density\n")
    
    # Ajouter l'événement
    with open(log_file, 'a') as f:
        f.write(f"{time_ms},{event_type},{count},{density:.6f}\n")


def save_adjacency_snapshot(syn_ee: Any, time_ms: float, outdir: str) -> None:
    """
    Sauvegarde un instantané de la matrice d'adjacence.
    
    Args:
        syn_ee: Objet Synapses Brian2 E→E
        time_ms: Temps de simulation
        outdir: Répertoire de sortie
    """
    import os
    
    # Reconstruction de la matrice d'adjacence
    N_e = syn_ee.source.N
    adj_matrix = np.zeros((N_e, N_e))
    
    for idx in range(len(syn_ee.i)):
        if syn_ee.alive[idx] > 0.5:  # Synapse active
            i_pre = syn_ee.i[idx]
            i_post = syn_ee.j[idx]
            adj_matrix[i_pre, i_post] = syn_ee.w[idx]
    
    # Sauvegarde
    filename = f"adjacency_t{int(time_ms)}.npy"
    filepath = os.path.join(outdir, filename)
    np.save(filepath, adj_matrix)


def compute_degree_histogram(syn_ee: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule l'histogramme des degrés entrants et sortants.
    
    Args:
        syn_ee: Objet Synapses Brian2 E→E
    
    Returns:
        (degrés_entrants, degrés_sortants) pour chaque neurone
    """
    N_e = syn_ee.source.N
    in_degree = np.zeros(N_e)
    out_degree = np.zeros(N_e)
    
    for idx in range(len(syn_ee.i)):
        if syn_ee.alive[idx] > 0.5:  # Synapse active
            i_pre = syn_ee.i[idx]
            i_post = syn_ee.j[idx]
            out_degree[i_pre] += 1
            in_degree[i_post] += 1
    
    return in_degree, out_degree 