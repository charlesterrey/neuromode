"""
Script d'exécution du modèle S-2 avec plasticité structurelle développementale.

Usage:
    python -m src.run_s2 --config configs/s2.json --outdir outputs/s2_run --plot
"""

import argparse
import os
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif

from .model_s2 import (
    load_config_s2, set_seeds_s2, build_network_s2, 
    create_structural_operation, run_sim_s2,
    save_results_s2, get_network_stats_s2
)
from .utils.io import (
    log_info, log_error, create_output_dir, save_json, ensure_dir
)


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Modèle S-2: LIF + STDP + scaling + plasticité structurelle"
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/s2.json',
        help='Chemin vers le fichier de configuration JSON'
    )
    
    parser.add_argument(
        '--outdir', 
        type=str,
        help='Répertoire de sortie (par défaut: outputs/run_TIMESTAMP)'
    )
    
    parser.add_argument(
        '--plot', 
        action='store_true',
        help='Générer les graphiques'
    )
    
    parser.add_argument(
        '--seed', 
        type=int,
        help='Graine aléatoire (override config)'
    )
    
    return parser.parse_args()


def create_plots_s2(monitors, cfg, outdir):
    """Génère les graphiques spécialisés S-2."""
    log_info("Génération des graphiques S-2...")
    
    plt.style.use('default')
    fig_dir = os.path.join(outdir, 'figures')
    ensure_dir(fig_dir)
    
    # === 1. DENSITÉ SYNAPTIQUE VS TEMPS ===
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if len(monitors['density_log']) > 0:
        density_data = np.array(monitors['density_log'])
        times = density_data[:, 0] / 1000  # en secondes
        densities = density_data[:, 1]
        
        ax.plot(times, densities, linewidth=2, color='darkblue', label='Densité synaptique')
        
        # Lignes de séparation des phases
        T_grow = cfg['struct']['phase']['T_grow_ms'] / 1000
        ax.axvline(x=T_grow, color='red', linestyle='--', alpha=0.7, 
                  label=f'Fin phase GROW ({T_grow:.1f}s)')
        
        # Annotation des phases
        ax.text(T_grow/2, max(densities)*0.9, 'PHASE GROW', 
               ha='center', va='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        total_duration = cfg['duration_ms'] / 1000
        if total_duration > T_grow:
            ax.text((T_grow + total_duration)/2, max(densities)*0.9, 'PHASE PRUNE', 
                   ha='center', va='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    ax.set_xlabel('Temps (s)')
    ax.set_ylabel('Densité synaptique')
    ax.set_title('Évolution de la densité synaptique - Phases développementales')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'density_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # === 2. RASTER PLOT E/I ===
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Spikes excitateurs
    if len(monitors['spikes_e'].i) > 0:
        ax1.scatter(monitors['spikes_e'].t/1000, monitors['spikes_e'].i, 
                   s=0.3, alpha=0.6, c='red', label='Excitateurs')
        ax1.set_ylabel('Neurone E')
        ax1.set_title('Raster des spikes')
        ax1.legend()
    
    # Spikes inhibiteurs  
    if len(monitors['spikes_i'].i) > 0:
        ax2.scatter(monitors['spikes_i'].t/1000, monitors['spikes_i'].i,
                   s=0.3, alpha=0.6, c='blue', label='Inhibiteurs')
        ax2.set_ylabel('Neurone I')
        ax2.set_xlabel('Temps (s)')
        ax2.legend()
    
    # Lignes de phase
    T_grow = cfg['struct']['phase']['T_grow_ms'] / 1000
    for ax in [ax1, ax2]:
        ax.axvline(x=T_grow, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'raster_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # === 3. TAUX DE DÉCHARGE E/I ===
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(monitors['rates_e'].t) > 0:
        ax.plot(monitors['rates_e'].t/1000, monitors['rates_e'].rate, 
               color='red', linewidth=2, label='Population E')
        
    if len(monitors['rates_i'].t) > 0:
        ax.plot(monitors['rates_i'].t/1000, monitors['rates_i'].rate,
               color='blue', linewidth=2, label='Population I')
    
    # Ligne de référence scaling
    if cfg['scaling']['enabled']:
        ax.axhline(y=cfg['scaling']['target_hz'], color='gray', 
                  linestyle='--', alpha=0.7, label=f"Cible: {cfg['scaling']['target_hz']} Hz")
    
    # Ligne de phase
    T_grow = cfg['struct']['phase']['T_grow_ms'] / 1000
    ax.axvline(x=T_grow, color='red', linestyle='--', alpha=0.7, label='Fin phase GROW')
    
    ax.set_xlabel('Temps (s)')
    ax.set_ylabel('Taux de décharge (Hz)')
    ax.set_title('Taux de décharge des populations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'population_rates.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # === 4. HISTOGRAMME DES DEGRÉS ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Calcul des degrés finaux
    from .plasticity.structural import compute_degree_histogram
    in_degrees, out_degrees = compute_degree_histogram(monitors['syn_ee'])
    
    # Degrés entrants
    ax1.hist(in_degrees, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Degré entrant')
    ax1.set_ylabel('Nombre de neurones')
    ax1.set_title('Distribution des degrés entrants')
    ax1.axvline(x=np.mean(in_degrees), color='red', linestyle='--', 
               label=f'Moyenne: {np.mean(in_degrees):.1f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Degrés sortants
    ax2.hist(out_degrees, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Degré sortant')
    ax2.set_ylabel('Nombre de neurones')
    ax2.set_title('Distribution des degrés sortants')
    ax2.axvline(x=np.mean(out_degrees), color='red', linestyle='--',
               label=f'Moyenne: {np.mean(out_degrees):.1f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'degree_histogram.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # === 5. TRAJECTOIRES POIDS/ACTIVITÉ ÉCHANTILLON ===
    if len(monitors['weights'].t) > 0:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        weight_times = monitors['weights'].t/1000  # en secondes
        weight_values = monitors['weights'].w
        A_values = monitors['weights'].act_score
        alive_values = monitors['weights'].alive
        
        # Ligne de phase
        T_grow = cfg['struct']['phase']['T_grow_ms'] / 1000
        for ax in [ax1, ax2, ax3]:
            ax.axvline(x=T_grow, color='red', linestyle='--', alpha=0.7)
        
        # Trajectoires des poids
        if weight_values.ndim > 1:
            n_show = min(10, weight_values.shape[1])
            indices_to_show = np.linspace(0, weight_values.shape[1]-1, n_show, dtype=int)
            
            for i in indices_to_show:
                if len(weight_times) == len(weight_values[:, i]):
                    ax1.plot(weight_times, weight_values[:, i], alpha=0.7, linewidth=1)
            
            # Trajectoires d'activité A
            for i in indices_to_show:
                if len(weight_times) == len(A_values[:, i]):
                    ax2.plot(weight_times, A_values[:, i], alpha=0.7, linewidth=1)
            
            # État alive (0/1)
            for i in indices_to_show:
                if len(weight_times) == len(alive_values[:, i]):
                    ax3.plot(weight_times, alive_values[:, i], alpha=0.7, linewidth=1)
        
        ax1.set_ylabel('Poids synaptique')
        ax1.set_title('Évolution des poids (échantillon)')
        ax1.axhline(y=cfg['syn']['w_max'], color='red', linestyle=':', alpha=0.5, label='w_max')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_ylabel('Score d\'activité A')
        ax2.set_title('Évolution du score d\'activité')
        ax2.axhline(y=cfg['struct']['theta_act'], color='red', linestyle=':', 
                   alpha=0.5, label=f'θ_act = {cfg["struct"]["theta_act"]}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.set_ylabel('État alive (0/1)')
        ax3.set_xlabel('Temps (s)')
        ax3.set_title('État des connexions (0=inactive, 1=active)')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'trajectories_sample.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    log_info(f"Graphiques S-2 sauvegardés dans {fig_dir}")


def main():
    """Fonction principale."""
    args = parse_args()
    
    try:
        # Chargement de la configuration
        log_info(f"Chargement de la configuration S-2: {args.config}")
        cfg = load_config_s2(args.config)
        
        # Override des paramètres via CLI
        if args.seed is not None:
            cfg['seed'] = args.seed
            log_info(f"Graine surchargée: {args.seed}")
        
        # Configuration des graines aléatoires
        set_seeds_s2(cfg['seed'])
        
        # Création du répertoire de sortie
        if args.outdir:
            outdir = args.outdir
        else:
            outdir = create_output_dir("outputs")
        ensure_dir(outdir)
        log_info(f"Répertoire de sortie: {outdir}")
        
        # Sauvegarde de la configuration effective
        config_path = os.path.join(outdir, 'config_effective.json')
        save_json(cfg, config_path)
        
        # Construction du réseau S-2
        net, monitors = build_network_s2(cfg)
        
        # Création de l'opération de plasticité structurelle
        structural_op = create_structural_operation(cfg, monitors, outdir)
        
        # Simulation avec phases développementales
        run_sim_s2(net, cfg['duration_ms'], cfg['scaling'], monitors, structural_op)
        
        # Calcul et affichage des statistiques
        stats = get_network_stats_s2(monitors)
        log_info("=== STATISTIQUES DU RÉSEAU S-2 ===")
        log_info(f"Spikes totaux E: {stats['total_spikes_e']}")
        log_info(f"Spikes totaux I: {stats['total_spikes_i']}")
        log_info(f"Taux moyen E: {stats['mean_rate_e']:.2f} Hz")
        log_info(f"Taux moyen I: {stats['mean_rate_i']:.2f} Hz")
        log_info(f"Connexions actives finales: {stats['active_connections']}")
        log_info(f"Densité finale: {stats['final_density']:.4f}")
        log_info(f"Densité max: {stats['max_density']:.4f}")
        log_info(f"Poids actifs - Moyenne: {stats['weight_mean_active']:.3f}")
        log_info(f"Poids actifs - Min/Max: {stats['weight_min_active']:.3f}/{stats['weight_max_active']:.3f}")
        
        # Vérifications de sanité
        if stats['has_nan_weights']:
            log_error("ATTENTION: Présence de NaN dans les poids!")
            return 1
        
        if stats['has_nan_activity']:
            log_error("ATTENTION: Présence de NaN dans les scores d'activité!")
            return 1
        
        if not stats['weights_in_bounds']:
            log_error("ATTENTION: Poids hors des bornes [0, w_max]!")
            return 1
        
        # Sauvegarde des résultats
        save_results_s2(monitors, cfg, outdir)
        
        # Sauvegarde des statistiques
        stats_path = os.path.join(outdir, 'network_stats.json')
        save_json(stats, stats_path)
        
        # Génération des graphiques
        if args.plot:
            create_plots_s2(monitors, cfg, outdir)
        
        # Vérification des critères d'acceptation
        log_info("=== VÉRIFICATION CRITÈRES D'ACCEPTATION ===")
        
        # 1. Densité suit hausse puis baisse
        if len(monitors['density_log']) > 0:
            density_data = np.array(monitors['density_log'])
            densities = density_data[:, 1]
            
            max_density_achieved = np.max(densities)
            final_density_achieved = densities[-1]
            
            if max_density_achieved > stats['final_density']:
                log_info("✓ Densité suit pattern hausse puis baisse")
            else:
                log_error("✗ Densité ne suit pas le pattern attendu")
        
        # 2. Réseau stable (FR raisonnables)
        if 0.1 <= stats['mean_rate_e'] <= 50.0 and 0.1 <= stats['mean_rate_i'] <= 50.0:
            log_info("✓ Réseau stable (taux de décharge raisonnables)")
        else:
            log_error(f"✗ Réseau instable (E: {stats['mean_rate_e']:.2f} Hz, I: {stats['mean_rate_i']:.2f} Hz)")
        
        # 3. Métriques produites
        expected_files = ['density_evolution.csv', 'degree_histogram.csv', 'structural_events.csv']
        all_files_exist = all(os.path.exists(os.path.join(outdir, f)) for f in expected_files)
        
        if all_files_exist:
            log_info("✓ Métriques/fichiers S-2 générés")
        else:
            log_error("✗ Fichiers manquants")
        
        log_info("=== SIMULATION S-2 TERMINÉE AVEC SUCCÈS ===")
        log_info(f"Résultats disponibles dans: {outdir}")
        
        return 0
        
    except Exception as e:
        log_error(f"Erreur lors de l'exécution S-2: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main()) 