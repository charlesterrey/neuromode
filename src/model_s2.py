"""
Modèle S-2 : Extension S-1 avec plasticité structurelle développementale.

Implémente la surcroissance suivie d'élagage activité-dépendant :
- Phase GROW : activation progressive de connexions E→E 
- Phase PRUNE : élagage basé sur activité (A) et coûts de câblage
- Règles locales STDP + scaling homéostatique conservées
- Alive gating : multiplication par variable 'alive' ∈ {0,1}
"""

import json
import os
import numpy as np
import pandas as pd
from brian2 import *
from typing import Dict, Any, Tuple, Optional
from .utils.io import log_info, log_warning, save_json, save_array
from .plasticity.structural import (
    compute_density, grow_step, prune_step, 
    create_position_grid, compute_distance_costs,
    log_structural_event, save_adjacency_snapshot,
    compute_degree_histogram
)


def load_config_s2(path: str) -> Dict[str, Any]:
    """Charge la configuration S-2 depuis un fichier JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def set_seeds_s2(seed_value: int) -> None:
    """Configure les graines aléatoires pour la reproductibilité."""
    np.random.seed(seed_value)
    seed(seed_value)  # Brian2 seed


def build_network_s2(cfg: Dict[str, Any]) -> Tuple[Network, Dict[str, Any]]:
    """
    Construit le réseau S-2 avec plasticité structurelle.
    
    Returns:
        net: Le réseau Brian2
        monitors: Dictionnaire des moniteurs + structures auxiliaires
    """
    log_info("Construction du réseau S-2...")
    
    # Avertissement pour grosses simulations
    N_e = cfg['N_e']
    if N_e > 800:
        log_warning(f"N_e={N_e} > 800 : préallocation full E→E peut être lourde en mémoire!")
    
    # Paramètres temporels
    defaultclock.dt = cfg['dt_ms'] * ms
    
    # Paramètres LIF (identiques S-1)
    tau_m = cfg['lif']['tau_m_ms'] * ms
    E_L = cfg['lif']['E_L_mV'] * mV
    V_th = cfg['lif']['V_th_mV'] * mV
    V_reset = cfg['lif']['V_reset_mV'] * mV
    refractory = cfg['lif']['refractory_ms'] * ms
    
    # Paramètres synaptiques
    tau_e = cfg['syn']['tau_e_ms'] * ms
    tau_i = cfg['syn']['tau_i_ms'] * ms
    
    # Équations LIF avec conductances (identiques S-1)
    lif_eq = """
    dv/dt = (E_L - v + g_e*(0*mV - v) + g_i*(-80*mV - v)) / tau_m : volt (unless refractory)
    dg_e/dt = -g_e / tau_e : 1
    dg_i/dt = -g_i / tau_i : 1
    """
    
    # Populations de neurones (identiques S-1)
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
    
    # === SYNAPSES E→E AVEC PLASTICITÉ STRUCTURELLE ===
    log_info("Création synapses E→E avec préallocation full et alive gating...")
    
    # Paramètres structuraux
    tau_A = cfg['struct']['tau_A_ms'] * ms
    beta_pre = cfg['struct']['beta_pre']
    beta_post = cfg['struct']['beta_post']
    
    # Paramètres STDP
    tau_pre = cfg['stdp']['tau_pre_ms'] * ms
    tau_post = cfg['stdp']['tau_post_ms'] * ms
    delta_Apre = cfg['stdp']['Apre']
    delta_Apost = cfg['stdp']['Apost']
    gmax_e = cfg['syn']['gmax_e']
    w_max = cfg['syn']['w_max']
    
    # Équations synaptiques E→E avec variables structurelles
    stdp_struct_eq = """
    w : 1
    act_score : 1  # Score d'activité pour élagage (géré manuellement)
    alive : 1  # Gating 0/1 pour connexion active
    len_cost : 1  # Coût de câblage normalisé
    dApre/dt = -Apre / tau_pre : 1 (event-driven)
    dApost/dt = -Apost / tau_post : 1 (event-driven)
    """
    
    # Actions pré- et post-synaptiques avec alive gating
    on_pre_struct = """
    g_e_post += alive * w * gmax_e
    Apre += delta_Apre
    w = clip(w + alive * Apost, 0, w_max)
    act_score += alive * beta_pre
    """
    
    on_post_struct = """
    Apost += delta_Apost  
    w = clip(w + alive * Apre, 0, w_max)
    act_score += alive * beta_post
    """
    
    # Création des synapses E→E avec connectivité full (sauf auto-connexions)
    syn_ee = Synapses(neurons_e, neurons_e, stdp_struct_eq,
                      on_pre=on_pre_struct, on_post=on_post_struct,
                      namespace={'tau_pre': tau_pre, 'tau_post': tau_post, 
                               'delta_Apre': delta_Apre, 'delta_Apost': delta_Apost,
                               'gmax_e': gmax_e, 'w_max': w_max, 'tau_A': tau_A,
                               'beta_pre': beta_pre, 'beta_post': beta_post})
    
    # Connexion full (tous-vers-tous sauf auto-connexions)
    syn_ee.connect(condition='i != j')
    
    log_info(f"Préallocation E→E : {len(syn_ee)} connexions potentielles")
    
    # === INITIALISATION DES VARIABLES STRUCTURELLES ===
    
    # Positions 2D des neurones pour calcul des coûts de câblage
    positions_e = create_position_grid(N_e, grid_size=10.0)
    
    # Calcul des coûts de câblage pour toutes les connexions possibles
    len_costs = compute_distance_costs(positions_e, positions_e, 
                                     lambda_len=cfg['struct']['lambda_len'])
    
    # Normalisation des coûts (0-1)
    if len(len_costs) > 0:
        len_costs = (len_costs - np.min(len_costs)) / (np.max(len_costs) - np.min(len_costs) + 1e-8)
    
    # Initialisation des variables synaptiques
    w_mean = cfg['syn']['w_init_mean']
    w_std = cfg['syn']['w_init_std']
    
    # Poids initiaux (toutes les connexions, même inactives)
    syn_ee.w = np.clip(np.random.normal(w_mean, w_std, len(syn_ee)), 0, w_max)
    
    # Initialisation alive : début avec quelques connexions actives
    initial_density = 0.05  # 5% de connexions actives au début
    n_initial = int(initial_density * len(syn_ee))
    syn_ee.alive = 0.0  # Toutes inactives au début
    initial_indices = np.random.choice(len(syn_ee), size=n_initial, replace=False)
    syn_ee.alive[initial_indices] = 1.0
    
    # Coûts de câblage
    syn_ee.len_cost = len_costs
    
    # Traces STDP et activité
    syn_ee.Apre = 0.0
    syn_ee.Apost = 0.0
    syn_ee.act_score = 0.0
    
    # === SYNAPSES AUTRES (identiques S-1 mais avec connectivité configurée) ===
    
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
    
    # === ENTRÉE EXTERNE (identique S-1) ===
    input_rate = 10 * Hz
    input_neurons = PoissonGroup(N_e // 4, rates=input_rate)
    syn_input = Synapses(input_neurons, neurons_e, 'w : 1', on_pre='g_e_post += w * gmax_e',
                        namespace={'gmax_e': gmax_e})
    syn_input.connect(p=0.3)
    syn_input.w = w_mean * 2
    
    # === MONITEURS ===
    
    spike_mon_e = SpikeMonitor(neurons_e)
    spike_mon_i = SpikeMonitor(neurons_i)
    
    rate_mon_e = PopulationRateMonitor(neurons_e)
    rate_mon_i = PopulationRateMonitor(neurons_i)
    
    # Échantillonnage pour monitoring des poids et activité
    from .plasticity.structural import sampling_indices_for_monitor
    n_sample = min(cfg['record']['sample_weights'], len(syn_ee))
    sample_indices = sampling_indices_for_monitor(len(syn_ee), n_sample)
    
    weight_mon = StateMonitor(syn_ee, ['w', 'act_score', 'alive'], record=sample_indices, dt=50*ms)
    
    # === ASSEMBLAGE DU RÉSEAU ===
    
    net = Network(neurons_e, neurons_i, syn_ee, syn_ei, syn_ie, syn_ii,
                  input_neurons, syn_input,
                  spike_mon_e, spike_mon_i, rate_mon_e, rate_mon_i, weight_mon)
    
    # Structures auxiliaires pour gestion développementale
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
        'rng': np.random.default_rng(cfg['seed']),  # RNG dédié pour structural
        'density_log': [],  # Log de l'évolution de la densité
        'structural_events': []  # Log des événements grow/prune
    }
    
    log_info(f"Réseau S-2 construit: {N_e} neurones E, {N_i} neurones I")
    log_info(f"Connexions E→E potentielles: {len(syn_ee)}")
    initial_density = compute_density(syn_ee.alive, N_e)
    log_info(f"Densité initiale: {initial_density:.4f}")
    
    return net, monitors


def create_structural_operation(cfg: Dict[str, Any], monitors: Dict[str, Any], 
                              outdir: str) -> Any:
    """
    Crée l'opération réseau pour la gestion de la plasticité structurelle.
    
    Returns:
        NetworkOperation Brian2 pour les phases GROW/PRUNE
    """
    
    # Sauvegarde de la config structurelle dans monitors pour accès ultérieur
    monitors['struct_cfg'] = cfg['struct']
    
    # Paramètres des phases
    T_grow = cfg['struct']['phase']['T_grow_ms']
    T_prune = cfg['struct']['phase']['T_prune_ms']
    total_duration = T_grow + T_prune
    
    # Paramètres structuraux
    rho_target_grow = cfg['struct']['rho_target_grow']
    max_add_per_step = cfg['struct']['max_add_per_step']
    theta_act = cfg['struct']['theta_act']
    k1 = cfg['struct']['k1']
    k2 = cfg['struct']['k2']
    
    syn_ee = monitors['syn_ee']
    N_e = cfg['N_e']
    rng = monitors['rng']
    
    # Compteurs d'événements
    grow_count = 0
    prune_count = 0
    
    @network_operation(dt=cfg['struct']['dt_struct_ms']*ms)
    def structural_plasticity():
        nonlocal grow_count, prune_count
        
        current_time_ms = float(defaultclock.t / ms)
        current_density = compute_density(syn_ee.alive, N_e)
        
        # Décroissance manuelle du score d'activité
        dt_struct_s = cfg['struct']['dt_struct_ms'] / 1000.0  # en secondes
        tau_A_s = cfg['struct']['tau_A_ms'] / 1000.0  # en secondes
        decay_factor = np.exp(-dt_struct_s / tau_A_s)
        syn_ee.act_score[:] *= decay_factor
        
        # Log de la densité
        monitors['density_log'].append((current_time_ms, current_density))
        
        if current_time_ms <= T_grow:
            # === PHASE GROW ===
            n_added = grow_step(syn_ee, rho_target_grow, max_add_per_step, rng)
            if n_added > 0:
                grow_count += n_added
                log_structural_event(current_time_ms, "GROW", n_added, 
                                   current_density, outdir)
                if grow_count % 1000 == 0:  # Log périodique
                    log_info(f"t={current_time_ms:.0f}ms GROW: +{n_added} synapses "
                           f"(total: {grow_count}, densité: {current_density:.4f})")
        
        elif current_time_ms <= total_duration:
            # === PHASE PRUNE ===
            n_pruned = prune_step(syn_ee, theta_act, k1, k2, rng)
            if n_pruned > 0:
                prune_count += n_pruned
                log_structural_event(current_time_ms, "PRUNE", n_pruned,
                                   current_density, outdir)
                if prune_count % 100 == 0:  # Log plus fréquent pour PRUNE
                    log_info(f"t={current_time_ms:.0f}ms PRUNE: -{n_pruned} synapses "
                           f"(total: {prune_count}, densité: {current_density:.4f})")
        
        # Sauvegarde périodique d'instantanés d'adjacence
        if current_time_ms % 1000 == 0:  # Tous les 1000ms
            save_adjacency_snapshot(syn_ee, current_time_ms, outdir)
    
    return structural_plasticity


def apply_homeostatic_scaling_s2(monitors: Dict[str, Any], cfg: Dict[str, Any], 
                                current_time_ms: float) -> None:
    """
    Applique le scaling homéostatique (identique S-1 mais adapté S-2).
    """
    if not cfg['scaling']['enabled']:
        return
        
    syn_ee = monitors['syn_ee']
    neurons_e = monitors['neurons_e']
    target_rate = cfg['scaling']['target_hz']
    eta = cfg['scaling']['eta_scale']
    min_scale = cfg['scaling']['min_scale']
    max_scale = cfg['scaling']['max_scale']
    
    # Calcul du taux récent
    interval_ms = cfg['scaling']['interval_ms']
    window_start = max(0, current_time_ms - interval_ms)
    
    spike_mon = monitors['spikes_e']
    recent_mask = (spike_mon.t/ms >= window_start) & (spike_mon.t/ms <= current_time_ms)
    recent_spikes = spike_mon.i[recent_mask]
    
    # Taux individuels
    spike_counts = np.bincount(recent_spikes, minlength=len(neurons_e))
    current_rates = spike_counts / (interval_ms / 1000.0)
    
    # Facteurs de scaling
    scaling_factors = np.ones(len(neurons_e))
    for post_idx in range(len(neurons_e)):
        if current_rates[post_idx] > 0:
            desired_factor = target_rate / current_rates[post_idx]
            scaling_factors[post_idx] = 1.0 + eta * (desired_factor - 1.0)
            scaling_factors[post_idx] = np.clip(scaling_factors[post_idx], 
                                              min_scale, max_scale)
    
    # Application aux poids des synapses ACTIVES seulement
    for syn_idx in range(len(syn_ee)):
        if syn_ee.alive[syn_idx] > 0.5:  # Synapse active
            post_neuron = syn_ee.j[syn_idx]
            syn_ee.w[syn_idx] *= scaling_factors[post_neuron]
            syn_ee.w[syn_idx] = np.clip(syn_ee.w[syn_idx], 0, cfg['syn']['w_max'])


def run_sim_s2(net: Network, duration_ms: float, scaling_cfg: Dict[str, Any],
              monitors: Dict[str, Any], structural_op: Any) -> None:
    """
    Exécute la simulation S-2 avec plasticité structurelle et scaling.
    """
    log_info(f"Démarrage simulation S-2 ({duration_ms} ms)...")
    
    # Récupération des paramètres structuraux depuis monitors (ajoutés dans create_structural_operation)
    struct_cfg = monitors.get('struct_cfg', {'phase': {'T_grow_ms': 3000}})
    log_info(f"Phase GROW: 0-{struct_cfg['phase']['T_grow_ms']} ms")
    log_info(f"Phase PRUNE: {struct_cfg['phase']['T_grow_ms']}-{duration_ms} ms")
    
    # Ajout de l'opération structurelle au réseau
    net.add(structural_op)
    
    if scaling_cfg['enabled']:
        interval_ms = scaling_cfg['interval_ms']
        n_intervals = int(duration_ms / interval_ms)
        
        for i in range(n_intervals):
            # Simulation d'un intervalle
            net.run(interval_ms * ms)
            
            # Application du scaling homéostatique
            current_time = (i + 1) * interval_ms
            apply_homeostatic_scaling_s2(monitors, {'scaling': scaling_cfg, 
                                                  'syn': {'w_max': 1.0}}, current_time)
            
            if (i + 1) % 10 == 0:  # Log moins fréquent
                log_info(f"Progression: {current_time:.0f}/{duration_ms} ms")
        
        # Temps restant
        remaining_ms = duration_ms - n_intervals * interval_ms
        if remaining_ms > 0:
            net.run(remaining_ms * ms)
    else:
        net.run(duration_ms * ms)
    
    log_info("Simulation S-2 terminée")


def save_results_s2(monitors: Dict[str, Any], cfg: Dict[str, Any], outdir: str) -> None:
    """Sauvegarde les résultats S-2 (étendu de S-1)."""
    log_info("Sauvegarde des résultats S-2...")
    
    # Spikes (identique S-1)
    spikes_e = np.column_stack([monitors['spikes_e'].i, monitors['spikes_e'].t/ms])
    spikes_i = np.column_stack([monitors['spikes_i'].i, monitors['spikes_i'].t/ms])
    
    max_spikes = cfg['record']['max_spikes']
    if len(spikes_e) > max_spikes:
        log_warning(f"Troncature spikes E: {len(spikes_e)} → {max_spikes}")
        spikes_e = spikes_e[:max_spikes]
    if len(spikes_i) > max_spikes:
        log_warning(f"Troncature spikes I: {len(spikes_i)} → {max_spikes}")
        spikes_i = spikes_i[:max_spikes]
    
    save_array("spikes_e", spikes_e, outdir, "csv")
    save_array("spikes_i", spikes_i, outdir, "csv")
    
    # Taux de population (identique S-1)
    rates_e = np.column_stack([monitors['rates_e'].t/ms, monitors['rates_e'].rate/Hz])
    rates_i = np.column_stack([monitors['rates_i'].t/ms, monitors['rates_i'].rate/Hz])
    save_array("rates_e", rates_e, outdir, "csv")
    save_array("rates_i", rates_i, outdir, "csv")
    
    # === DONNÉES SPÉCIFIQUES S-2 ===
    
    # Trajectoires poids, activité et alive des échantillons
    weight_times = monitors['weights'].t/ms
    weight_values = monitors['weights'].w
    A_values = monitors['weights'].act_score
    alive_values = monitors['weights'].alive
    
    # Sauvegarde des trajectoires (format étendu)
    if weight_values.ndim > 1:
        w_data = np.column_stack([weight_times, weight_values.T])
        A_data = np.column_stack([weight_times, A_values.T])
        alive_data = np.column_stack([weight_times, alive_values.T])
    else:
        w_data = np.column_stack([weight_times, weight_values])
        A_data = np.column_stack([weight_times, A_values])
        alive_data = np.column_stack([weight_times, alive_values])
    
    # Sauvegarde en numpy pour efficacité
    np.save(os.path.join(outdir, "weight_trajectories.npy"), w_data)
    np.save(os.path.join(outdir, "activity_trajectories.npy"), A_data)
    np.save(os.path.join(outdir, "alive_trajectories.npy"), alive_data)
    
    # États finaux de toutes les synapses E→E
    final_weights = np.array(monitors['syn_ee'].w)
    final_A = np.array(monitors['syn_ee'].act_score)
    final_alive = np.array(monitors['syn_ee'].alive)
    final_len_cost = np.array(monitors['syn_ee'].len_cost)
    
    np.save(os.path.join(outdir, "final_weights_ee.npy"), final_weights)
    np.save(os.path.join(outdir, "final_activity_ee.npy"), final_A)
    np.save(os.path.join(outdir, "final_alive_ee.npy"), final_alive)
    np.save(os.path.join(outdir, "final_len_costs_ee.npy"), final_len_cost)
    
    # Évolution de la densité synaptique
    density_data = np.array(monitors['density_log'])
    if len(density_data) > 0:
        save_array("density_evolution", density_data, outdir, "csv")
    
    # Histogrammes des degrés finaux
    in_degrees, out_degrees = compute_degree_histogram(monitors['syn_ee'])
    degree_data = np.column_stack([in_degrees, out_degrees])
    save_array("degree_histogram", degree_data, outdir, "csv")
    
    log_info(f"Résultats S-2 sauvegardés dans {outdir}")


def get_network_stats_s2(monitors: Dict[str, Any]) -> Dict[str, Any]:
    """Calcule les statistiques étendues du réseau S-2."""
    stats = {}
    
    # Statistiques de base (identiques S-1)
    stats['total_spikes_e'] = len(monitors['spikes_e'].i)
    stats['total_spikes_i'] = len(monitors['spikes_i'].i)
    
    if len(monitors['rates_e'].rate) > 0:
        stats['mean_rate_e'] = float(np.mean(monitors['rates_e'].rate/Hz))
        stats['mean_rate_i'] = float(np.mean(monitors['rates_i'].rate/Hz))
    else:
        stats['mean_rate_e'] = 0.0
        stats['mean_rate_i'] = 0.0
    
    # Statistiques des connexions actives
    syn_ee = monitors['syn_ee']
    active_mask = syn_ee.alive > 0.5
    active_weights = np.array(syn_ee.w)[active_mask]
    
    if len(active_weights) > 0:
        stats['active_connections'] = int(np.sum(active_mask))
        stats['weight_mean_active'] = float(np.mean(active_weights))
        stats['weight_std_active'] = float(np.std(active_weights))
        stats['weight_min_active'] = float(np.min(active_weights))
        stats['weight_max_active'] = float(np.max(active_weights))
    else:
        stats['active_connections'] = 0
        stats['weight_mean_active'] = 0.0
        stats['weight_std_active'] = 0.0
        stats['weight_min_active'] = 0.0
        stats['weight_max_active'] = 0.0
    
    # Densité synaptique finale
    N_e = syn_ee.source.N
    stats['final_density'] = compute_density(syn_ee.alive, N_e)
    
    # Évolution de la densité
    if len(monitors['density_log']) > 0:
        density_evolution = np.array(monitors['density_log'])
        stats['max_density'] = float(np.max(density_evolution[:, 1]))
        stats['min_density'] = float(np.min(density_evolution[:, 1]))
    else:
        stats['max_density'] = stats['final_density']
        stats['min_density'] = stats['final_density']
    
    # Vérifications de sanité
    all_weights = np.array(syn_ee.w)
    all_A = np.array(syn_ee.act_score)
    
    stats['has_nan_weights'] = bool(np.any(np.isnan(all_weights)))
    stats['has_nan_activity'] = bool(np.any(np.isnan(all_A)))
    stats['weights_in_bounds'] = bool(np.all((all_weights >= 0) & (all_weights <= 1.0)))
    
    return stats 