"""
Modèle S-3 : Extension S-2 avec fenêtres critiques γ(t) et contraintes énergétiques.

Ajoute la modulation développementale de la plasticité via γ(t) et
les pressions métaboliques via budget énergétique pour l'élagage.
"""

import json
import os
import numpy as np
import pandas as pd
from brian2 import *
from typing import Dict, Any, Tuple, Optional

# Import des modules S-2 comme base
from .model_s2 import (
    set_seeds_s2, apply_homeostatic_scaling_s2, 
    save_results_s2, get_network_stats_s2
)
from .utils.io import log_info, log_warning, save_json, save_array
from .utils.gating import create_critical_window_scheduler
from .utils.energy import create_energy_tracker, compute_synapse_activity_count, compute_total_wiring_cost
from .plasticity.structural import (
    compute_density, grow_step, sigmoid,
    create_position_grid, compute_distance_costs,
    log_structural_event, save_adjacency_snapshot,
    compute_degree_histogram, sampling_indices_for_monitor
)


def load_config_s3(path: str) -> Dict[str, Any]:
    """Charge la configuration S-3 depuis un fichier JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def set_seeds_s3(seed_value: int) -> None:
    """Configure les graines aléatoires pour la reproductibilité S-3."""
    np.random.seed(seed_value)
    seed(seed_value)  # Brian2 seed


def build_network_s3(cfg: Dict[str, Any]) -> Tuple[Network, Dict[str, Any]]:
    """
    Construit le réseau S-3 avec modulation γ(t) et contraintes énergétiques.
    
    Returns:
        net: Le réseau Brian2
        monitors: Dictionnaire des moniteurs + structures auxiliaires S-3
    """
    log_info("Construction du réseau S-3...")
    
    # Avertissement pour grosses simulations (identique S-2)
    N_e = cfg['N_e']
    if N_e > 800:
        log_warning(f"N_e={N_e} > 800 : préallocation full E→E peut être lourde en mémoire!")
    
    # Paramètres temporels
    defaultclock.dt = cfg['dt_ms'] * ms
    
    # Paramètres LIF (identiques S-2)
    tau_m = cfg['lif']['tau_m_ms'] * ms
    E_L = cfg['lif']['E_L_mV'] * mV
    V_th = cfg['lif']['V_th_mV'] * mV
    V_reset = cfg['lif']['V_reset_mV'] * mV
    refractory = cfg['lif']['refractory_ms'] * ms
    
    # Paramètres synaptiques
    tau_e = cfg['syn']['tau_e_ms'] * ms
    tau_i = cfg['syn']['tau_i_ms'] * ms
    
    # Équations LIF (identiques S-2)
    lif_eq = """
    dv/dt = (E_L - v + g_e*(0*mV - v) + g_i*(-80*mV - v)) / tau_m : volt (unless refractory)
    dg_e/dt = -g_e / tau_e : 1
    dg_i/dt = -g_i / tau_i : 1
    """
    
    # Populations de neurones (identiques S-2)
    N_i = cfg['N_i']
    
    neurons_e = NeuronGroup(N_e, lif_eq,
                           threshold='v > V_th',
                           reset='v = V_reset',
                           refractory=refractory,
                           method='euler',
                           namespace={'tau_m': tau_m, 'E_L': E_L, 'V_th': V_th, 
                                    'V_reset': V_reset, 'tau_e': tau_e, 'tau_i': tau_i})
    neurons_e.v = E_L + (V_th - E_L) * np.random.random(N_e)
    neurons_e.g_e = 0
    neurons_e.g_i = 0
    
    neurons_i = NeuronGroup(N_i, lif_eq,
                           threshold='v > V_th',
                           reset='v = V_reset',
                           refractory=refractory,
                           method='euler',
                           namespace={'tau_m': tau_m, 'E_L': E_L, 'V_th': V_th, 
                                    'V_reset': V_reset, 'tau_e': tau_e, 'tau_i': tau_i})
    neurons_i.v = E_L + (V_th - E_L) * np.random.random(N_i)
    neurons_i.g_e = 0
    neurons_i.g_i = 0
    
    # === SYNAPSES E→E AVEC MODULATION γ(t) ===
    log_info("Création synapses E→E avec modulation γ(t) et alive gating...")
    
    # Paramètres structuraux (identiques S-2)
    tau_A = cfg['struct']['tau_A_ms'] * ms
    beta_pre = cfg['struct']['beta_pre']
    beta_post = cfg['struct']['beta_post']
    
    # Paramètres STDP avec modulation γ(t)
    tau_pre = cfg['stdp']['tau_pre_ms'] * ms
    tau_post = cfg['stdp']['tau_post_ms'] * ms
    Apre_base = cfg['stdp']['Apre_base']
    Apost_base = cfg['stdp']['Apost_base']
    gmax_e = cfg['syn']['gmax_e']
    w_max = cfg['syn']['w_max']
    
    # Équations synaptiques E→E avec variables γ(t) (extension S-2)
    stdp_struct_eq_s3 = """
    w : 1
    act_score : 1  # Score d'activité pour élagage
    alive : 1  # Gating 0/1 pour connexion active
    len_cost : 1  # Coût de câblage normalisé
    gamma_factor : 1  # Facteur γ(t) mis à jour dynamiquement
    dApre/dt = -Apre / tau_pre : 1 (event-driven)
    dApost/dt = -Apost / tau_post : 1 (event-driven)
    """
    
    # Actions STDP modulées par γ(t)
    on_pre_struct_s3 = """
    g_e_post += alive * w * gmax_e
    Apre += gamma_factor * Apre_base
    w = clip(w + alive * gamma_factor * Apost, 0, w_max)
    act_score += alive * beta_pre
    """
    
    on_post_struct_s3 = """
    Apost += gamma_factor * Apost_base  
    w = clip(w + alive * gamma_factor * Apre, 0, w_max)
    act_score += alive * beta_post
    """
    
    # Création des synapses E→E (identique S-2 + γ(t))
    syn_ee = Synapses(neurons_e, neurons_e, stdp_struct_eq_s3,
                      on_pre=on_pre_struct_s3, on_post=on_post_struct_s3,
                      namespace={'tau_pre': tau_pre, 'tau_post': tau_post, 
                               'Apre_base': Apre_base, 'Apost_base': Apost_base,
                               'gmax_e': gmax_e, 'w_max': w_max, 'tau_A': tau_A,
                               'beta_pre': beta_pre, 'beta_post': beta_post})
    
    # Connexion full (identique S-2)
    syn_ee.connect(condition='i != j')
    log_info(f"Préallocation E→E : {len(syn_ee)} connexions potentielles")
    
    # === INITIALISATION DES VARIABLES (identique S-2 + γ) ===
    
    # Positions et coûts de câblage (identiques S-2)
    positions_e = create_position_grid(N_e, grid_size=10.0)
    len_costs = compute_distance_costs(positions_e, positions_e, 
                                     lambda_len=cfg['struct']['lambda_len'])
    if len(len_costs) > 0:
        len_costs = (len_costs - np.min(len_costs)) / (np.max(len_costs) - np.min(len_costs) + 1e-8)
    
    # Initialisation des variables synaptiques (identiques S-2)
    w_mean = cfg['syn']['w_init_mean']
    w_std = cfg['syn']['w_init_std']
    
    syn_ee.w = np.clip(np.random.normal(w_mean, w_std, len(syn_ee)), 0, w_max)
    
    initial_density = 0.05
    n_initial = int(initial_density * len(syn_ee))
    syn_ee.alive = 0.0
    initial_indices = np.random.choice(len(syn_ee), size=n_initial, replace=False)
    syn_ee.alive[initial_indices] = 1.0
    
    syn_ee.len_cost = len_costs
    syn_ee.Apre = 0.0
    syn_ee.Apost = 0.0
    syn_ee.act_score = 0.0
    syn_ee.gamma_factor = 1.0  # Initialisation γ(t)
    
    # === SYNAPSES AUTRES (identiques S-2) ===
    
    p_connect_EI = cfg['p_connect_EI']
    
    # E→I (fixes)
    syn_ei = Synapses(neurons_e, neurons_i, 'w : 1', on_pre='g_e_post += w * gmax_e',
                      namespace={'gmax_e': gmax_e})
    syn_ei.connect(p=p_connect_EI)
    syn_ei.w = w_mean
    
    # I→E (fixes, inhibitrices)
    gmax_i = cfg['syn']['gmax_i']
    syn_ie = Synapses(neurons_i, neurons_e, 'w : 1', on_pre='g_i_post += w * gmax_i',
                      namespace={'gmax_i': gmax_i})
    syn_ie.connect(p=p_connect_EI)
    syn_ie.w = w_mean
    
    # I→I (fixes, inhibitrices)
    syn_ii = Synapses(neurons_i, neurons_i, 'w : 1', on_pre='g_i_post += w * gmax_i',
                      namespace={'gmax_i': gmax_i})
    syn_ii.connect(p=p_connect_EI)
    syn_ii.w = w_mean
    
    # === ENTRÉE EXTERNE (identique S-2) ===
    input_rate = 10 * Hz
    input_neurons = PoissonGroup(N_e // 4, rates=input_rate)
    syn_input = Synapses(input_neurons, neurons_e, 'w : 1', on_pre='g_e_post += w * gmax_e',
                        namespace={'gmax_e': gmax_e})
    syn_input.connect(p=0.3)
    syn_input.w = w_mean * 2
    
    # === MONITEURS (étendus S-3) ===
    
    spike_mon_e = SpikeMonitor(neurons_e)
    spike_mon_i = SpikeMonitor(neurons_i)
    
    rate_mon_e = PopulationRateMonitor(neurons_e)
    rate_mon_i = PopulationRateMonitor(neurons_i)
    
    # Échantillonnage pour monitoring
    n_sample = min(cfg['record']['sample_weights'], len(syn_ee))
    sample_indices = sampling_indices_for_monitor(len(syn_ee), n_sample)
    
    weight_mon = StateMonitor(syn_ee, ['w', 'act_score', 'alive', 'gamma_factor'], 
                             record=sample_indices, dt=50*ms)
    
    # === ASSEMBLAGE DU RÉSEAU ===
    
    net = Network(neurons_e, neurons_i, syn_ee, syn_ei, syn_ie, syn_ii,
                  input_neurons, syn_input,
                  spike_mon_e, spike_mon_i, rate_mon_e, rate_mon_i, weight_mon)
    
    # === STRUCTURES AUXILIAIRES S-3 ===
    
    # Planificateur γ(t)
    gamma_scheduler = create_critical_window_scheduler(cfg['gamma'])
    
    # Tracker énergétique
    energy_tracker = create_energy_tracker(cfg['energy'])
    
    monitors = {
        'spikes_e': spike_mon_e,
        'spikes_i': spike_mon_i,
        'rates_e': rate_mon_e,
        'rates_i': rate_mon_i,
        'weights': weight_mon,
        'sample_indices': sample_indices,
        'syn_ee': syn_ee,
        'neurons_e': neurons_e,
        'neurons_i': neurons_i,
        'positions_e': positions_e,
        'rng': np.random.default_rng(cfg['seed']),
        'density_log': [],
        'structural_events': [],
        
        # Nouveaux éléments S-3
        'gamma_scheduler': gamma_scheduler,
        'energy_tracker': energy_tracker,
        'gamma_log': [],  # Log de γ(t)
        'energy_log': [],  # Log énergétique
        'struct_cfg': cfg['struct']  # Pour compatibilité
    }
    
    log_info(f"Réseau S-3 construit: {N_e} neurones E, {N_i} neurones I")
    log_info(f"Connexions E→E potentielles: {len(syn_ee)}")
    initial_density = compute_density(syn_ee.alive, N_e)
    log_info(f"Densité initiale: {initial_density:.4f}")
    log_info(f"Planificateur γ(t): {gamma_scheduler.get_description()}")
    
    return net, monitors


def prune_step_s3(syn_ee: Any, theta_act: float, k1: float, k2: float, kE: float,
                 gamma_current: float, pressure_E: float, rng: np.random.Generator, 
                 reset_A_fraction: float = 0.5) -> int:
    """
    Étape d'élagage S-3 : activité + coût câblage + pression énergétique.
    
    Formule: p_prune = σ(γ*k1*(θ_act - A) + k2*len_cost + kE*P_E)
    
    Args:
        syn_ee: Objet Synapses Brian2 E→E
        theta_act: Seuil d'activité pour l'élagage
        k1: Poids du terme d'activité
        k2: Poids du coût de câblage  
        kE: Poids de la pression énergétique
        gamma_current: Valeur actuelle de γ(t)
        pressure_E: Pression énergétique P_E
        rng: Générateur de nombres aléatoires
        reset_A_fraction: Fraction de A à réinitialiser après élagage
    
    Returns:
        Nombre de synapses élaguées
    """
    # Trouver les synapses actives
    active_indices = np.where(syn_ee.alive == 1)[0]
    
    if len(active_indices) == 0:
        return 0
    
    # Calcul des probabilités de pruning S-3
    A_active = syn_ee.act_score[active_indices]
    len_cost_active = syn_ee.len_cost[active_indices]
    
    # Score de pruning avec modulation γ(t) et pression énergétique
    prune_score = (gamma_current * k1 * (theta_act - A_active) + 
                  k2 * len_cost_active + 
                  kE * pressure_E)
    
    p_prune = sigmoid(prune_score, k=1.0)
    
    # Tirage aléatoire pour élagage
    random_vals = rng.random(len(active_indices))
    to_prune_mask = random_vals < p_prune
    to_prune = active_indices[to_prune_mask]
    
    if len(to_prune) == 0:
        return 0
    
    # Élagage des synapses sélectionnées
    syn_ee.alive[to_prune] = 0.0
    syn_ee.w[to_prune] = 0.0
    
    # Réinitialisation partielle des scores d'activité
    if reset_A_fraction > 0:
        surviving_indices = np.where(syn_ee.alive == 1)[0]
        if len(surviving_indices) > 0:
            decay_factor = 1.0 - reset_A_fraction
            syn_ee.act_score[surviving_indices] *= decay_factor
    
    return len(to_prune)


def create_structural_operation_s3(cfg: Dict[str, Any], monitors: Dict[str, Any], 
                                  outdir: str) -> Any:
    """
    Crée l'opération réseau S-3 avec γ(t) et contraintes énergétiques.
    
    Returns:
        NetworkOperation Brian2 pour plasticité S-3
    """
    
    # Paramètres (identiques S-2 + S-3)
    T_grow = cfg['struct']['phase']['T_grow_ms']
    T_prune = cfg['struct']['phase']['T_prune_ms'] 
    total_duration = T_grow + T_prune
    
    rho_target_grow = cfg['struct']['rho_target_grow']
    max_add_per_step = cfg['struct']['max_add_per_step']
    theta_act = cfg['struct']['theta_act']
    k1 = cfg['struct']['k1']
    k2 = cfg['struct']['k2']
    kE = cfg['struct']['kE']  # Nouveau paramètre S-3
    
    syn_ee = monitors['syn_ee']
    N_e = cfg['N_e']
    rng = monitors['rng']
    
    # Objets S-3
    gamma_scheduler = monitors['gamma_scheduler']
    energy_tracker = monitors['energy_tracker']
    
    # Compteurs
    grow_count = 0
    prune_count = 0
    
    @network_operation(dt=cfg['struct']['dt_struct_ms']*ms)
    def structural_plasticity_s3():
        nonlocal grow_count, prune_count
        
        current_time_ms = float(defaultclock.t / ms)
        current_density = compute_density(syn_ee.alive, N_e)
        
        # === MISE À JOUR γ(t) ===
        gamma_current = gamma_scheduler.value(current_time_ms)
        
        # Mise à jour du facteur γ dans toutes les synapses E→E
        syn_ee.gamma_factor[:] = gamma_current
        
        # Log γ(t)
        monitors['gamma_log'].append((current_time_ms, gamma_current))
        
        # === MISE À JOUR ÉNERGÉTIQUE ===
        
        # Compter les spikes récents
        dt_interval = cfg['struct']['dt_struct_ms']
        recent_spikes_e = 0
        if len(monitors['spikes_e'].t) > 0:
            recent_spike_times = monitors['spikes_e'].t / ms  # en ms
            recent_spikes_e = np.sum(recent_spike_times >= (current_time_ms - dt_interval))
        
        # Compter les événements synaptiques
        syn_events = compute_synapse_activity_count(monitors, dt_interval)
        
        # Compter le coût total de câblage
        total_wiring = compute_total_wiring_cost(monitors)
        
        # Mise à jour du tracker énergétique
        E_win, P_E = energy_tracker.update(current_time_ms, recent_spikes_e, syn_events, total_wiring)
        
        # Log énergétique
        monitors['energy_log'].append((current_time_ms, E_win, P_E, energy_tracker.budget_B))
        
        # === DÉCROISSANCE MANUELLE DU SCORE D'ACTIVITÉ ===
        dt_struct_s = cfg['struct']['dt_struct_ms'] / 1000.0
        tau_A_s = cfg['struct']['tau_A_ms'] / 1000.0
        decay_factor = np.exp(-dt_struct_s / tau_A_s)
        syn_ee.act_score[:] *= decay_factor
        
        # Log de la densité
        monitors['density_log'].append((current_time_ms, current_density))
        
        # === PHASES STRUCTURELLES ===
        
        if current_time_ms <= T_grow:
            # Phase GROW (identique S-2)
            n_added = grow_step(syn_ee, rho_target_grow, max_add_per_step, rng)
            if n_added > 0:
                grow_count += n_added
                log_structural_event(current_time_ms, "GROW", n_added, 
                                   current_density, outdir)
                if grow_count % 1000 == 0:
                    log_info(f"t={current_time_ms:.0f}ms GROW: +{n_added} synapses "
                           f"(total: {grow_count}, densité: {current_density:.4f}, γ={gamma_current:.3f})")
        
        elif current_time_ms <= total_duration:
            # Phase PRUNE S-3 avec γ(t) et P_E
            n_pruned = prune_step_s3(syn_ee, theta_act, k1, k2, kE, 
                                    gamma_current, P_E, rng)
            if n_pruned > 0:
                prune_count += n_pruned
                log_structural_event(current_time_ms, "PRUNE", n_pruned,
                                   current_density, outdir)
                if prune_count % 100 == 0:
                    log_info(f"t={current_time_ms:.0f}ms PRUNE: -{n_pruned} synapses "
                           f"(total: {prune_count}, densité: {current_density:.4f}, "
                           f"γ={gamma_current:.3f}, P_E={P_E:.3f})")
        
        # Sauvegarde périodique d'instantanés
        if current_time_ms % 1000 == 0:
            save_adjacency_snapshot(syn_ee, current_time_ms, outdir)
    
    return structural_plasticity_s3


def save_results_s3(monitors: Dict[str, Any], cfg: Dict[str, Any], outdir: str) -> None:
    """Sauvegarde les résultats S-3 (étendu de S-2 avec γ et énergie)."""
    log_info("Sauvegarde des résultats S-3...")
    
    # Sauvegarde des données de base S-2
    save_results_s2(monitors, cfg, outdir)
    
    # === DONNÉES SPÉCIFIQUES S-3 ===
    
    # Log γ(t)
    if len(monitors['gamma_log']) > 0:
        gamma_data = np.array(monitors['gamma_log'])
        save_array("gamma", gamma_data, outdir, "csv")
    
    # Log énergétique  
    if len(monitors['energy_log']) > 0:
        energy_data = np.array(monitors['energy_log'])
        # Colonnes: time_ms, E_win, P_E, budget
        energy_df = pd.DataFrame(energy_data, columns=['time_ms', 'E_win', 'P_E', 'budget'])
        energy_df.to_csv(os.path.join(outdir, 'energy.csv'), index=False)
    
    # Sauvegarde tracker énergétique complet
    energy_tracker = monitors['energy_tracker']
    energy_tracker.save_to_csv(os.path.join(outdir, 'energy_detailed.csv'))
    
    log_info(f"Résultats S-3 sauvegardés dans {outdir}")


def get_network_stats_s3(monitors: Dict[str, Any]) -> Dict[str, Any]:
    """Calcule les statistiques étendues du réseau S-3."""
    stats = get_network_stats_s2(monitors)  # Base S-2
    
    # === STATISTIQUES S-3 ===
    
    # Statistiques γ(t)
    if len(monitors['gamma_log']) > 0:
        gamma_data = np.array(monitors['gamma_log'])
        stats['gamma_mean'] = float(np.mean(gamma_data[:, 1]))
        stats['gamma_max'] = float(np.max(gamma_data[:, 1]))
        stats['gamma_final'] = float(gamma_data[-1, 1])
    else:
        stats['gamma_mean'] = 1.0
        stats['gamma_max'] = 1.0  
        stats['gamma_final'] = 1.0
    
    # Statistiques énergétiques
    if len(monitors['energy_log']) > 0:
        energy_data = np.array(monitors['energy_log'])
        E_wins = energy_data[:, 1]
        P_Es = energy_data[:, 2]
        budgets = energy_data[:, 3]
        
        stats['energy_mean'] = float(np.mean(E_wins))
        stats['energy_max'] = float(np.max(E_wins))
        stats['pressure_mean'] = float(np.mean(P_Es))
        stats['pressure_max'] = float(np.max(P_Es))
        stats['budget_exceeded_fraction'] = float(np.mean(E_wins > budgets[0]))
    else:
        stats['energy_mean'] = 0.0
        stats['energy_max'] = 0.0
        stats['pressure_mean'] = 0.0
        stats['pressure_max'] = 0.0
        stats['budget_exceeded_fraction'] = 0.0
    
    return stats 