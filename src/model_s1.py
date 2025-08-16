"""
Modèle S-1 : Réseau LIF avec STDP et scaling homéostatique.

Implémentation basée sur le plan neurobiologique :
- Neurones Leaky Integrate-and-Fire (LIF) avec conductances excitatrices/inhibitrices
- Plasticité synaptique STDP (Spike-Timing Dependent Plasticity) pair-based sur les connexions E→E
- Scaling homéostatique pour maintenir l'activité proche d'un niveau cible
- Pas encore de pruning ni de neuromodulation LC-NE (réservés pour S-2/S-3)
"""

import json
import os
import numpy as np
import pandas as pd
from brian2 import *
from typing import Dict, Any, Tuple, Optional
from .utils.io import log_info, log_warning, save_json, save_array


def load_config(path: str) -> Dict[str, Any]:
    """Charge la configuration depuis un fichier JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def set_seeds(seed_value: int) -> None:
    """Configure les graines aléatoires pour la reproductibilité."""
    np.random.seed(seed_value)
    seed(seed_value)  # Brian2 seed


def build_network(cfg: Dict[str, Any]) -> Tuple[Network, Dict[str, Any]]:
    """
    Construit le réseau neuronal avec LIF + STDP + scaling.
    
    Returns:
        net: Le réseau Brian2
        monitors: Dictionnaire des moniteurs pour l'analyse
    """
    log_info("Construction du réseau...")
    
    # Paramètres temporels
    defaultclock.dt = cfg['dt_ms'] * ms
    
    # Paramètres LIF
    tau_m = cfg['lif']['tau_m_ms'] * ms
    E_L = cfg['lif']['E_L_mV'] * mV
    V_th = cfg['lif']['V_th_mV'] * mV
    V_reset = cfg['lif']['V_reset_mV'] * mV
    refractory = cfg['lif']['refractory_ms'] * ms
    
    # Paramètres synaptiques
    tau_e = cfg['syn']['tau_e_ms'] * ms
    tau_i = cfg['syn']['tau_i_ms'] * ms
    
    # Équations LIF avec conductances excitatrices et inhibitrices
    lif_eq = """
    dv/dt = (E_L - v + g_e*(0*mV - v) + g_i*(-80*mV - v)) / tau_m : volt (unless refractory)
    dg_e/dt = -g_e / tau_e : 1
    dg_i/dt = -g_i / tau_i : 1
    """
    
    # Population excitatrice (E)
    N_e = cfg['N_e']
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
    
    # Population inhibitrice (I) 
    N_i = cfg['N_i']
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
    
    # Probabilité de connexion
    p_connect = cfg['p_connect']
    
    # Synapses E→E avec STDP
    log_info("Création des synapses E→E avec STDP...")
    stdp_eq = """
    w : 1
    dApre/dt = -Apre / tau_pre : 1 (event-driven)
    dApost/dt = -Apost / tau_post : 1 (event-driven)
    """
    
    on_pre_stdp = """
    g_e_post += w * gmax_e
    Apre += delta_Apre
    w = clip(w + Apost, 0, w_max)
    """
    
    on_post_stdp = """
    Apost += delta_Apost
    w = clip(w + Apre, 0, w_max)
    """
    
    # Paramètres STDP
    tau_pre = cfg['stdp']['tau_pre_ms'] * ms
    tau_post = cfg['stdp']['tau_post_ms'] * ms
    delta_Apre = cfg['stdp']['Apre']
    delta_Apost = cfg['stdp']['Apost']
    gmax_e = cfg['syn']['gmax_e']
    w_max = cfg['syn']['w_max']
    
    syn_ee = Synapses(neurons_e, neurons_e, stdp_eq,
                      on_pre=on_pre_stdp, on_post=on_post_stdp,
                      namespace={'tau_pre': tau_pre, 'tau_post': tau_post, 
                               'delta_Apre': delta_Apre, 'delta_Apost': delta_Apost,
                               'gmax_e': gmax_e, 'w_max': w_max})
    syn_ee.connect(p=p_connect)
    
    # Initialisation des poids E→E
    w_mean = cfg['syn']['w_init_mean']
    w_std = cfg['syn']['w_init_std']
    syn_ee.w = np.clip(np.random.normal(w_mean, w_std, len(syn_ee)), 0, w_max)
    syn_ee.Apre = 0
    syn_ee.Apost = 0
    
    # Synapses E→I (fixes)
    syn_ei = Synapses(neurons_e, neurons_i, 'w : 1', on_pre='g_e_post += w * gmax_e',
                      namespace={'gmax_e': gmax_e})
    syn_ei.connect(p=p_connect)
    syn_ei.w = w_mean
    
    # Synapses I→E (fixes, inhibitrices)
    gmax_i = cfg['syn']['gmax_i']
    syn_ie = Synapses(neurons_i, neurons_e, 'w : 1', on_pre='g_i_post += w * gmax_i',
                      namespace={'gmax_i': gmax_i})
    syn_ie.connect(p=p_connect)
    syn_ie.w = w_mean
    
    # Synapses I→I (fixes, inhibitrices)
    syn_ii = Synapses(neurons_i, neurons_i, 'w : 1', on_pre='g_i_post += w * gmax_i',
                      namespace={'gmax_i': gmax_i})
    syn_ii.connect(p=p_connect)
    syn_ii.w = w_mean
    
    # Entrée externe pour stimuler le réseau
    input_rate = 10 * Hz  # Taux d'entrée externe
    input_neurons = PoissonGroup(N_e // 4, rates=input_rate)  # 25% des neurones reçoivent de l'entrée
    syn_input = Synapses(input_neurons, neurons_e, 'w : 1', on_pre='g_e_post += w * gmax_e',
                        namespace={'gmax_e': gmax_e})
    syn_input.connect(p=0.3)  # Connexion plus dense pour l'entrée
    syn_input.w = w_mean * 2  # Poids plus forts pour l'entrée
    
    # Moniteurs
    spike_mon_e = SpikeMonitor(neurons_e)
    spike_mon_i = SpikeMonitor(neurons_i)
    
    rate_mon_e = PopulationRateMonitor(neurons_e)
    rate_mon_i = PopulationRateMonitor(neurons_i)
    
    # Échantillonnage des poids E→E pour suivi temporel
    n_sample = min(cfg['record']['sample_weights'], len(syn_ee))
    sample_indices = np.random.choice(len(syn_ee), n_sample, replace=False)
    weight_mon = StateMonitor(syn_ee, 'w', record=sample_indices, dt=10*ms)
    
    # Assemblage du réseau
    net = Network(neurons_e, neurons_i, syn_ee, syn_ei, syn_ie, syn_ii,
                  input_neurons, syn_input,
                  spike_mon_e, spike_mon_i, rate_mon_e, rate_mon_i, weight_mon)
    
    monitors = {
        'spikes_e': spike_mon_e,
        'spikes_i': spike_mon_i,
        'rates_e': rate_mon_e,
        'rates_i': rate_mon_i,
        'weights': weight_mon,
        'sample_indices': sample_indices
    }
    
    # Stocker les références pour le scaling
    monitors['syn_ee'] = syn_ee
    monitors['neurons_e'] = neurons_e
    
    log_info(f"Réseau construit: {N_e} neurones E, {N_i} neurones I")
    log_info(f"Connexions E→E: {len(syn_ee)} synapses avec STDP")
    
    return net, monitors


def apply_homeostatic_scaling(monitors: Dict[str, Any], cfg: Dict[str, Any], 
                            current_time_ms: float) -> None:
    """
    Applique le scaling homéostatique aux poids entrants par neurone.
    
    Ajuste les poids pour maintenir le firing rate proche du niveau cible.
    """
    if not cfg['scaling']['enabled']:
        return
        
    syn_ee = monitors['syn_ee']
    neurons_e = monitors['neurons_e']
    target_rate = cfg['scaling']['target_hz']
    eta = cfg['scaling']['eta_scale']
    min_scale = cfg['scaling']['min_scale']
    max_scale = cfg['scaling']['max_scale']
    
    # Calcul du taux de décharge récent pour chaque neurone E
    interval_ms = cfg['scaling']['interval_ms']
    window_start = max(0, current_time_ms - interval_ms)
    
    # Récupération des spikes dans la fenêtre récente
    spike_mon = monitors['spikes_e']
    recent_mask = (spike_mon.t/ms >= window_start) & (spike_mon.t/ms <= current_time_ms)
    recent_spikes = spike_mon.i[recent_mask]
    
    # Calcul des taux individuels
    spike_counts = np.bincount(recent_spikes, minlength=len(neurons_e))
    current_rates = spike_counts / (interval_ms / 1000.0)  # Hz
    
    # Facteurs de scaling par neurone postsynaptique
    scaling_factors = np.ones(len(neurons_e))
    for post_idx in range(len(neurons_e)):
        if current_rates[post_idx] > 0:
            # Facteur multiplicatif pour rapprocher du taux cible
            desired_factor = target_rate / current_rates[post_idx]
            scaling_factors[post_idx] = 1.0 + eta * (desired_factor - 1.0)
            scaling_factors[post_idx] = np.clip(scaling_factors[post_idx], 
                                              min_scale, max_scale)
    
    # Application du scaling aux poids entrants de chaque neurone
    for syn_idx in range(len(syn_ee)):
        post_neuron = syn_ee.j[syn_idx]
        syn_ee.w[syn_idx] *= scaling_factors[post_neuron]
        syn_ee.w[syn_idx] = np.clip(syn_ee.w[syn_idx], 0, cfg['syn']['w_max'])


def run_sim(net: Network, duration_ms: float, scaling_cfg: Dict[str, Any],
           monitors: Dict[str, Any]) -> None:
    """
    Exécute la simulation avec scaling homéostatique périodique.
    """
    log_info(f"Démarrage de la simulation ({duration_ms} ms)...")
    
    if scaling_cfg['enabled']:
        interval_ms = scaling_cfg['interval_ms']
        n_intervals = int(duration_ms / interval_ms)
        
        for i in range(n_intervals):
            # Simulation d'un intervalle
            net.run(interval_ms * ms)
            
            # Application du scaling homéostatique
            current_time = (i + 1) * interval_ms
            apply_homeostatic_scaling(monitors, {'scaling': scaling_cfg, 
                                               'syn': {'w_max': 1.0}}, current_time)
            
            if (i + 1) % 5 == 0:  # Log tous les 5 intervalles
                log_info(f"Progression: {current_time:.0f}/{duration_ms} ms")
        
        # Simulation du temps restant
        remaining_ms = duration_ms - n_intervals * interval_ms
        if remaining_ms > 0:
            net.run(remaining_ms * ms)
    else:
        net.run(duration_ms * ms)
    
    log_info("Simulation terminée")


def save_results(monitors: Dict[str, Any], cfg: Dict[str, Any], outdir: str) -> None:
    """Sauvegarde les résultats de la simulation."""
    log_info("Sauvegarde des résultats...")
    
    # Spikes
    spikes_e = np.column_stack([monitors['spikes_e'].i, monitors['spikes_e'].t/ms])
    spikes_i = np.column_stack([monitors['spikes_i'].i, monitors['spikes_i'].t/ms])
    
    max_spikes = cfg['record']['max_spikes']
    if len(spikes_e) > max_spikes:
        log_warning(f"Troncature des spikes E: {len(spikes_e)} → {max_spikes}")
        spikes_e = spikes_e[:max_spikes]
    if len(spikes_i) > max_spikes:
        log_warning(f"Troncature des spikes I: {len(spikes_i)} → {max_spikes}")
        spikes_i = spikes_i[:max_spikes]
    
    save_array("spikes_e", spikes_e, outdir, "csv")
    save_array("spikes_i", spikes_i, outdir, "csv")
    
    # Taux de population
    rates_e = np.column_stack([monitors['rates_e'].t/ms, monitors['rates_e'].rate/Hz])
    rates_i = np.column_stack([monitors['rates_i'].t/ms, monitors['rates_i'].rate/Hz])
    save_array("rates_e", rates_e, outdir, "csv")
    save_array("rates_i", rates_i, outdir, "csv")
    
    # Poids échantillonnés  
    weight_times = monitors['weights'].t/ms
    weight_values = monitors['weights'].w
    
    # Sauvegarde des trajectoires temporelles des poids
    # Note: weight_values est de forme (nb_synapses, nb_temps), il faut transposer
    if weight_values.ndim > 1:
        # Transposer pour avoir (nb_temps, nb_synapses)
        weight_values_T = weight_values.T
        columns = ['time_ms'] + [f'weight_{i}' for i in range(weight_values_T.shape[1])]
        weight_data = np.column_stack([weight_times, weight_values_T])
    else:
        # Un seul poids échantillonné
        columns = ['time_ms', 'weight_0']
        weight_data = np.column_stack([weight_times, weight_values])
    
    # Sauvegarder directement en tant que DataFrame
    weight_df = pd.DataFrame(weight_data, columns=columns)
    weight_df.to_csv(os.path.join(outdir, "weight_trajectories.csv"), index=False)
    
    # Poids finaux de toutes les synapses E→E
    final_weights = np.array(monitors['syn_ee'].w)
    save_array("final_weights_ee", final_weights, outdir, "npy")
    
    log_info(f"Résultats sauvegardés dans {outdir}")


def get_network_stats(monitors: Dict[str, Any]) -> Dict[str, Any]:
    """Calcule les statistiques du réseau."""
    stats = {}
    
    # Statistiques des spikes
    stats['total_spikes_e'] = len(monitors['spikes_e'].i)
    stats['total_spikes_i'] = len(monitors['spikes_i'].i)
    
    # Taux moyens
    if len(monitors['rates_e'].rate) > 0:
        stats['mean_rate_e'] = float(np.mean(monitors['rates_e'].rate/Hz))
        stats['mean_rate_i'] = float(np.mean(monitors['rates_i'].rate/Hz))
    else:
        stats['mean_rate_e'] = 0.0
        stats['mean_rate_i'] = 0.0
    
    # Statistiques des poids
    final_weights = np.array(monitors['syn_ee'].w)
    stats['weight_mean'] = float(np.mean(final_weights))
    stats['weight_std'] = float(np.std(final_weights))
    stats['weight_min'] = float(np.min(final_weights))
    stats['weight_max'] = float(np.max(final_weights))
    
    # Vérification de NaN
    stats['has_nan_weights'] = bool(np.any(np.isnan(final_weights)))
    
    return stats 