"""
Modèle S-4 : Extension S-3 avec offloading Ω(t) et contrôle LC-NE g_NE(t).

Ajoute la variable d'offloading mémoire et la modulation neuromodulateur
LC-NE qui réduisent l'effort endogène et la plasticité synaptique.
"""

import json
import os
import numpy as np
import pandas as pd
from brian2 import *
from typing import Dict, Any, Tuple, Optional

# Import des modules S-3 comme base
from .model_s3 import (
    set_seeds_s3, apply_homeostatic_scaling_s2, 
    save_results_s3, get_network_stats_s3,
    create_structural_operation_s3, prune_step_s3
)
from .utils.io import log_info, log_warning, save_json, save_array
from .utils.gating import create_critical_window_scheduler
from .utils.energy import create_energy_tracker
from .utils.offloading import create_offloading_scheduler, create_ne_controller, compute_odi
from .plasticity.structural import (
    compute_density, grow_step, sigmoid,
    create_position_grid, compute_distance_costs,
    log_structural_event, save_adjacency_snapshot,
    compute_degree_histogram, sampling_indices_for_monitor
)


def load_config_s4(path: str) -> Dict[str, Any]:
    """Charge la configuration S-4 depuis un fichier JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def set_seeds_s4(seed_value: int) -> None:
    """Configure les graines aléatoires pour la reproductibilité S-4."""
    np.random.seed(seed_value)
    seed(seed_value)  # Brian2 seed


def build_network_s4(cfg: Dict[str, Any]) -> Tuple[Network, Dict[str, Any]]:
    """
    Construit le réseau S-4 avec offloading Ω(t) et modulation LC-NE.
    
    Returns:
        net: Le réseau Brian2
        monitors: Dictionnaire des moniteurs + structures auxiliaires S-4
    """
    log_info("Construction du réseau S-4...")
    
    # Validation t0 dans durée simulation
    t0_ms = cfg['offloading']['t0_ms']
    duration_ms = cfg['duration_ms']
    if t0_ms < 0 or t0_ms > duration_ms:
        log_warning(f"t0={t0_ms}ms hors de [0,{duration_ms}ms] !")
    
    # Réutilisation construction S-3 comme base
    from .model_s3 import build_network_s3
    net, monitors = build_network_s3(cfg)
    
    # === STRUCTURES SUPPLÉMENTAIRES S-4 ===
    
    # Planificateur d'offloading Ω(t)
    offloading_schedule = create_offloading_scheduler(cfg['offloading'])
    
    # Contrôleur LC-NE g_NE(t)
    ne_controller = create_ne_controller(cfg['lc_ne'], seed=cfg['seed'])
    
    # Logs supplémentaires S-4
    monitors.update({
        'offloading_schedule': offloading_schedule,
        'ne_controller': ne_controller,
        'effort_log': [],      # Log effort(t)
        'gne_log': [],         # Log g_NE(t)
    })
    
    log_info(f"Réseau S-4 construit avec {offloading_schedule.get_description()}")
    log_info(f"Contrôleur: {ne_controller.get_description()}")
    
    return net, monitors


def create_structural_operation_s4(cfg: Dict[str, Any], monitors: Dict[str, Any], 
                                  outdir: str) -> Any:
    """
    Crée l'opération réseau S-4 avec Ω(t), γ(t), LC-NE et contraintes énergétiques.
    
    Returns:
        NetworkOperation Brian2 pour plasticité S-4
    """
    
    # Paramètres (identiques S-3)
    T_grow = cfg['struct']['phase']['T_grow_ms']
    T_prune = cfg['struct']['phase']['T_prune_ms'] 
    total_duration = T_grow + T_prune
    
    rho_target_grow = cfg['struct']['rho_target_grow']
    max_add_per_step = cfg['struct']['max_add_per_step']
    theta_act = cfg['struct']['theta_act']
    k1 = cfg['struct']['k1']
    k2 = cfg['struct']['k2']
    kE = cfg['struct']['kE']
    
    syn_ee = monitors['syn_ee']
    N_e = cfg['N_e']
    rng = monitors['rng']
    
    # Objets S-4
    gamma_scheduler = monitors['gamma_scheduler']
    energy_tracker = monitors['energy_tracker']
    offloading_schedule = monitors['offloading_schedule']
    ne_controller = monitors['ne_controller']
    
    # Compteurs
    grow_count = 0
    prune_count = 0
    
    @network_operation(dt=cfg['struct']['dt_struct_ms']*ms)
    def structural_plasticity_s4():
        nonlocal grow_count, prune_count
        
        current_time_ms = float(defaultclock.t / ms)
        current_density = compute_density(syn_ee.alive, N_e)
        
        # === MISE À JOUR γ(t) (identique S-3) ===
        gamma_current = gamma_scheduler.value(current_time_ms)
        
        # === MISE À JOUR OFFLOADING ET LC-NE (NOUVEAU S-4) ===
        effort_current = offloading_schedule.effort(current_time_ms)
        dt_interval = cfg['struct']['dt_struct_ms']
        gne_current = ne_controller.value(current_time_ms, effort_current, dt_interval)
        
        # Modulation conjointe γ(t) * g_NE(t) sur les synapses E→E
        combined_modulation = gamma_current * gne_current
        syn_ee.gamma_factor[:] = combined_modulation
        
        # Logs S-4
        monitors['effort_log'].append((current_time_ms, effort_current))
        monitors['gne_log'].append((current_time_ms, gne_current))
        monitors['gamma_log'].append((current_time_ms, gamma_current))
        
        # === MISE À JOUR ÉNERGÉTIQUE (identique S-3) ===
        recent_spikes_e = 0
        if len(monitors['spikes_e'].t) > 0:
            recent_spike_times = monitors['spikes_e'].t / ms
            recent_spikes_e = np.sum(recent_spike_times >= (current_time_ms - dt_interval))
        
        from .utils.energy import compute_synapse_activity_count, compute_total_wiring_cost
        syn_events = compute_synapse_activity_count(monitors, dt_interval)
        total_wiring = compute_total_wiring_cost(monitors)
        
        E_win, P_E = energy_tracker.update(current_time_ms, recent_spikes_e, syn_events, total_wiring)
        monitors['energy_log'].append((current_time_ms, E_win, P_E, energy_tracker.budget_B))
        
        # === DÉCROISSANCE MANUELLE DU SCORE D'ACTIVITÉ ===
        dt_struct_s = cfg['struct']['dt_struct_ms'] / 1000.0
        tau_A_s = cfg['struct']['tau_A_ms'] / 1000.0
        decay_factor = np.exp(-dt_struct_s / tau_A_s)
        syn_ee.act_score[:] *= decay_factor
        
        # Log de la densité
        monitors['density_log'].append((current_time_ms, current_density))
        
        # === PHASES STRUCTURELLES (identique S-3) ===
        
        if current_time_ms <= T_grow:
            # Phase GROW
            n_added = grow_step(syn_ee, rho_target_grow, max_add_per_step, rng)
            if n_added > 0:
                grow_count += n_added
                log_structural_event(current_time_ms, "GROW", n_added, 
                                   current_density, outdir)
                if grow_count % 1000 == 0:
                    log_info(f"t={current_time_ms:.0f}ms GROW: +{n_added} synapses "
                           f"(total: {grow_count}, densité: {current_density:.4f}, "
                           f"γ={gamma_current:.3f}, g_NE={gne_current:.3f}, effort={effort_current:.3f})")
        
        elif current_time_ms <= total_duration:
            # Phase PRUNE S-4 avec γ(t), g_NE(t) et P_E
            n_pruned = prune_step_s3(syn_ee, theta_act, k1, k2, kE, 
                                    combined_modulation, P_E, rng)
            if n_pruned > 0:
                prune_count += n_pruned
                log_structural_event(current_time_ms, "PRUNE", n_pruned,
                                   current_density, outdir)
                if prune_count % 100 == 0:
                    log_info(f"t={current_time_ms:.0f}ms PRUNE: -{n_pruned} synapses "
                           f"(total: {prune_count}, densité: {current_density:.4f}, "
                           f"γ={gamma_current:.3f}, g_NE={gne_current:.3f}, effort={effort_current:.3f}, P_E={P_E:.3f})")
        
        # Sauvegarde périodique d'instantanés
        if current_time_ms % 1000 == 0:
            save_adjacency_snapshot(syn_ee, current_time_ms, outdir)
    
    return structural_plasticity_s4


def run_probe(net: Network, monitors: Dict[str, Any], cfg: Dict[str, Any], 
              assist_mode: bool = False) -> float:
    """
    Lance une sonde de rappel simplifiée pour évaluer la performance du réseau.
    
    Args:
        net: Réseau Brian2
        monitors: Moniteurs
        cfg: Configuration
        assist_mode: Si True, utilise une stimulation assistée
        
    Returns:
        Score de rappel (proxy basé sur l'activité récente)
    """
    probe_cfg = cfg['probe']
    
    # Mesure simplifiée de l'activité récente
    if len(monitors['spikes_e'].t) == 0:
        return 0.0
    
    # Activité des dernières 200ms comme proxy
    current_time = defaultclock.t / ms
    recent_window = probe_cfg.get('baseline_window_ms', 200)
    recent_spikes = monitors['spikes_e'].t / ms
    recent_mask = recent_spikes >= (current_time - recent_window)
    recent_activity = np.sum(recent_mask)
    
    # Modulation selon le mode assist
    if assist_mode:
        # Mode assisté : activité stimulée (proxy)
        stim_factor = 2.0
    else:
        # Mode non-assisté : activité réduite (proxy)  
        stim_factor = 0.5
    
    # Score de rappel proxy basé sur l'activité récente modulée
    baseline_rate = recent_activity / (recent_window / 1000.0)
    recall_score = baseline_rate * stim_factor
    
    return recall_score


def save_results_s4(monitors: Dict[str, Any], cfg: Dict[str, Any], outdir: str) -> None:
    """Sauvegarde les résultats S-4 (étendu de S-3 avec offloading et LC-NE)."""
    log_info("Sauvegarde des résultats S-4...")
    
    # Sauvegarde des données de base S-3
    save_results_s3(monitors, cfg, outdir)
    
    # === DONNÉES SPÉCIFIQUES S-4 ===
    
    # Log effort(t)
    if len(monitors['effort_log']) > 0:
        effort_data = np.array(monitors['effort_log'])
        save_array("effort", effort_data, outdir, "csv")
    
    # Log g_NE(t)  
    if len(monitors['gne_log']) > 0:
        gne_data = np.array(monitors['gne_log'])
        save_array("gne", gne_data, outdir, "csv")
    
    log_info(f"Résultats S-4 sauvegardés dans {outdir}")


def get_network_stats_s4(monitors: Dict[str, Any]) -> Dict[str, Any]:
    """Calcule les statistiques étendues du réseau S-4."""
    stats = get_network_stats_s3(monitors)  # Base S-3
    
    # === STATISTIQUES S-4 ===
    
    # Statistiques effort(t)
    if len(monitors['effort_log']) > 0:
        effort_data = np.array(monitors['effort_log'])
        stats['effort_mean'] = float(np.mean(effort_data[:, 1]))
        stats['effort_min'] = float(np.min(effort_data[:, 1]))
        stats['effort_final'] = float(effort_data[-1, 1])
    else:
        stats['effort_mean'] = 1.0
        stats['effort_min'] = 1.0
        stats['effort_final'] = 1.0
    
    # Statistiques g_NE(t)
    if len(monitors['gne_log']) > 0:
        gne_data = np.array(monitors['gne_log'])
        stats['gne_mean'] = float(np.mean(gne_data[:, 1]))
        stats['gne_max'] = float(np.max(gne_data[:, 1]))
        stats['gne_final'] = float(gne_data[-1, 1])
    else:
        stats['gne_mean'] = 1.0
        stats['gne_max'] = 1.0
        stats['gne_final'] = 1.0
    
    return stats 