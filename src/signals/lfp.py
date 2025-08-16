"""
Module de construction de signaux LFP-like multi-canaux.

Convertit les trains de spikes Brian2 en signaux pseudo-LFP
pour l'analyse de connectivité directionnelle (PDC/dDTF).
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from scipy import signal


def bin_spikes_to_rate(spike_times_ms: np.ndarray, spike_indices: np.ndarray, 
                       n_neurons: int, dt_bin_ms: float = 2.0) -> Tuple[np.ndarray, float]:
    """
    Binnage des spikes en taux de décharge par fenêtre temporelle.
    
    Args:
        spike_times_ms: Temps des spikes en ms
        spike_indices: Indices des neurones qui ont émis les spikes
        n_neurons: Nombre total de neurones
        dt_bin_ms: Taille des bins en ms
        
    Returns:
        rate_matrix: Matrice (T, n_neurons) des taux de décharge
        fs: Fréquence d'échantillonnage (Hz)
    """
    if len(spike_times_ms) == 0:
        # Cas sans spikes
        duration_ms = 1000.0  # Durée par défaut
        n_bins = int(duration_ms / dt_bin_ms)
        return np.zeros((n_bins, n_neurons)), 1000.0 / dt_bin_ms
    
    # Paramètres temporels
    t_min = np.min(spike_times_ms)
    t_max = np.max(spike_times_ms)
    duration_ms = t_max - t_min
    
    n_bins = int(np.ceil(duration_ms / dt_bin_ms))
    fs = 1000.0 / dt_bin_ms  # Fréquence d'échantillonnage en Hz
    
    # Matrice de taux (temps x neurones)
    rate_matrix = np.zeros((n_bins, n_neurons))
    
    # Binnage par neurone
    for neuron_id in range(n_neurons):
        neuron_spikes = spike_times_ms[spike_indices == neuron_id]
        if len(neuron_spikes) > 0:
            # Histogramme des spikes
            bins = np.linspace(t_min, t_max, n_bins + 1)
            counts, _ = np.histogram(neuron_spikes, bins=bins)
            
            # Conversion en taux (spikes/s)
            rate_matrix[:, neuron_id] = counts / (dt_bin_ms / 1000.0)
    
    return rate_matrix, fs


def alpha_kernel(t: np.ndarray, tau_ms: float) -> np.ndarray:
    """
    Noyau alpha pour convolution des taux de décharge.
    
    Formule: k(t) = (t/τ) * exp(-t/τ) pour t >= 0
    
    Args:
        t: Vecteur temps en ms
        tau_ms: Constante de temps en ms
        
    Returns:
        Noyau alpha normalisé
    """
    tau_s = tau_ms / 1000.0  # Conversion en secondes
    kernel = np.zeros_like(t)
    
    # Noyau alpha pour t >= 0
    positive_mask = t >= 0
    t_pos = t[positive_mask]
    kernel[positive_mask] = (t_pos / tau_s) * np.exp(-t_pos / tau_s)
    
    # Normalisation (aire = 1)
    if np.sum(kernel) > 0:
        kernel = kernel / np.sum(kernel)
    
    return kernel


def convolve_alpha(rate_matrix: np.ndarray, tau_ms: float, fs: float) -> np.ndarray:
    """
    Convolution des taux avec un noyau alpha pour lisser les signaux.
    
    Args:
        rate_matrix: Matrice (T, K) des taux de décharge
        tau_ms: Constante de temps du noyau alpha
        fs: Fréquence d'échantillonnage
        
    Returns:
        Signaux lissés (T, K)
    """
    T, K = rate_matrix.shape
    
    # Construction du noyau alpha
    dt = 1.0 / fs  # Pas temporel en secondes
    kernel_duration = 5 * tau_ms  # 5 constantes de temps
    n_kernel = int(kernel_duration * fs / 1000.0)
    
    t_kernel = np.arange(n_kernel) * dt * 1000.0  # en ms
    kernel = alpha_kernel(t_kernel, tau_ms)
    
    # Convolution canal par canal
    smoothed = np.zeros_like(rate_matrix)
    for k in range(K):
        smoothed[:, k] = np.convolve(rate_matrix[:, k], kernel, mode='same')
    
    return smoothed


def cluster_neurons(n_neurons: int, n_groups: int, method: str = 'sequential') -> List[List[int]]:
    """
    Regroupe les neurones en clusters pour former des canaux LFP.
    
    Args:
        n_neurons: Nombre total de neurones
        n_groups: Nombre de groupes désirés
        method: Méthode de clustering ('sequential', 'random')
        
    Returns:
        Liste de listes d'indices de neurones par groupe
    """
    neuron_indices = np.arange(n_neurons)
    
    if method == 'sequential':
        # Groupement séquentiel
        group_size = n_neurons // n_groups
        groups = []
        
        for g in range(n_groups):
            start_idx = g * group_size
            if g == n_groups - 1:
                # Dernier groupe prend les neurones restants
                end_idx = n_neurons
            else:
                end_idx = (g + 1) * group_size
            
            groups.append(neuron_indices[start_idx:end_idx].tolist())
            
    elif method == 'random':
        # Groupement aléatoire
        shuffled = np.random.permutation(neuron_indices)
        groups = np.array_split(shuffled, n_groups)
        groups = [group.tolist() for group in groups]
        
    else:
        raise ValueError(f"Méthode de clustering non supportée: {method}")
    
    return groups


def make_lfp_proxy(spike_times_ms: np.ndarray, spike_indices: np.ndarray,
                   n_neurons: int, n_groups: int = 20, dt_bin_ms: float = 2.0,
                   tau_ms: float = 15.0, method: str = 'alpha',
                   cluster_method: str = 'sequential') -> Tuple[np.ndarray, float, List[List[int]]]:
    """
    Construction de signaux LFP-like multi-canaux à partir des spikes.
    
    Pipeline: spikes → bins → groupement → lissage → signaux LFP proxy
    
    Args:
        spike_times_ms: Temps des spikes en ms
        spike_indices: Indices des neurones
        n_neurons: Nombre total de neurones
        n_groups: Nombre de canaux LFP (groupes de neurones)
        dt_bin_ms: Résolution temporelle des bins
        tau_ms: Constante de temps pour lissage
        method: Méthode de lissage ('alpha', 'simple')
        cluster_method: Méthode de clustering des neurones
        
    Returns:
        lfp_signals: Signaux LFP (T, K) où K = n_groups
        fs: Fréquence d'échantillonnage
        groups: Groupes de neurones pour chaque canal
    """
    # Étape 1: Binnage des spikes en taux
    rate_matrix, fs = bin_spikes_to_rate(spike_times_ms, spike_indices, 
                                       n_neurons, dt_bin_ms)
    
    # Étape 2: Clustering des neurones
    groups = cluster_neurons(n_neurons, n_groups, cluster_method)
    
    # Étape 3: Agrégation par groupe
    T = rate_matrix.shape[0]
    lfp_signals = np.zeros((T, n_groups))
    
    for g, neuron_group in enumerate(groups):
        if len(neuron_group) > 0:
            # Moyenne des taux du groupe
            lfp_signals[:, g] = np.mean(rate_matrix[:, neuron_group], axis=1)
    
    # Étape 4: Lissage selon la méthode
    if method == 'alpha':
        lfp_signals = convolve_alpha(lfp_signals, tau_ms, fs)
    elif method == 'simple':
        # Lissage simple par moyenne mobile
        window_size = max(1, int(tau_ms * fs / 1000.0))
        for k in range(n_groups):
            lfp_signals[:, k] = np.convolve(lfp_signals[:, k], 
                                          np.ones(window_size)/window_size, 
                                          mode='same')
    
    # Étape 5: Normalisation z-score par canal
    for k in range(n_groups):
        signal_k = lfp_signals[:, k]
        if np.std(signal_k) > 1e-6:  # Éviter division par zéro
            lfp_signals[:, k] = (signal_k - np.mean(signal_k)) / np.std(signal_k)
    
    return lfp_signals, fs, groups


def segment_signals(signals: np.ndarray, segment_length: int, 
                   overlap: float = 0.5) -> List[np.ndarray]:
    """
    Segmente les signaux en fenêtres pour analyse stationnarité.
    
    Args:
        signals: Signaux (T, K)
        segment_length: Longueur des segments
        overlap: Fraction de recouvrement [0,1)
        
    Returns:
        Liste des segments (chaque segment: (segment_length, K))
    """
    T, K = signals.shape
    step = int(segment_length * (1 - overlap))
    
    segments = []
    start = 0
    
    while start + segment_length <= T:
        segment = signals[start:start + segment_length, :]
        segments.append(segment)
        start += step
    
    return segments


def preprocess_for_var(signals: np.ndarray, detrend: bool = True) -> np.ndarray:
    """
    Préparation des signaux pour analyse VAR.
    
    Args:
        signals: Signaux bruts (T, K)
        detrend: Si True, applique un detrending linéaire
        
    Returns:
        Signaux préprocessés
    """
    T, K = signals.shape
    processed = signals.copy()
    
    # Detrending linéaire par canal
    if detrend:
        for k in range(K):
            processed[:, k] = signal.detrend(processed[:, k], type='linear')
    
    # Re-normalisation après detrending
    for k in range(K):
        signal_k = processed[:, k]
        if np.std(signal_k) > 1e-6:
            processed[:, k] = (signal_k - np.mean(signal_k)) / np.std(signal_k)
    
    return processed 