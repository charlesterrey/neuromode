"""
Script d'exécution du modèle S-1 avec interface en ligne de commande.

Usage:
    python -m src.run_s1 --config configs/default.json --duration_ms 5000 --plot
"""

import argparse
import os
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour éviter les problèmes d'affichage

from .model_s1 import (
    load_config, set_seeds, build_network, run_sim, 
    save_results, get_network_stats
)
from .utils.io import (
    log_info, log_error, create_output_dir, save_json, ensure_dir
)


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Modèle S-1: LIF + STDP + scaling homéostatique"
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/default.json',
        help='Chemin vers le fichier de configuration JSON'
    )
    
    parser.add_argument(
        '--duration_ms', 
        type=float,
        help='Durée de simulation en ms (override config)'
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


def create_plots(monitors, cfg, outdir):
    """Génère les graphiques d'analyse."""
    log_info("Génération des graphiques...")
    
    plt.style.use('default')
    fig_dir = os.path.join(outdir, 'figures')
    ensure_dir(fig_dir)
    
    # 1. Raster plot des spikes E et I
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Spikes excitateurs
    if len(monitors['spikes_e'].i) > 0:
        ax1.scatter(monitors['spikes_e'].t/1000, monitors['spikes_e'].i, 
                   s=0.5, alpha=0.6, c='red', label='Excitateurs')
        ax1.set_ylabel('Neurone E')
        ax1.set_title('Raster des spikes')
        ax1.legend()
    
    # Spikes inhibiteurs  
    if len(monitors['spikes_i'].i) > 0:
        ax2.scatter(monitors['spikes_i'].t/1000, monitors['spikes_i'].i,
                   s=0.5, alpha=0.6, c='blue', label='Inhibiteurs')
        ax2.set_ylabel('Neurone I')
        ax2.set_xlabel('Temps (s)')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'raster_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Taux de décharge des populations
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(monitors['rates_e'].t) > 0:
        ax.plot(monitors['rates_e'].t/1000, monitors['rates_e'].rate, 
               color='red', linewidth=2, label='Population E')
        
    if len(monitors['rates_i'].t) > 0:
        ax.plot(monitors['rates_i'].t/1000, monitors['rates_i'].rate,
               color='blue', linewidth=2, label='Population I')
    
    # Ligne de référence pour le taux cible
    if cfg['scaling']['enabled']:
        ax.axhline(y=cfg['scaling']['target_hz'], color='gray', 
                  linestyle='--', alpha=0.7, label=f"Cible: {cfg['scaling']['target_hz']} Hz")
    
    ax.set_xlabel('Temps (s)')
    ax.set_ylabel('Taux de décharge (Hz)')
    ax.set_title('Taux de décharge des populations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'population_rates.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Distribution des poids initiaux vs finaux
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Poids initiaux (approximation gaussienne)
    w_init_samples = np.clip(
        np.random.normal(cfg['syn']['w_init_mean'], cfg['syn']['w_init_std'], 10000),
        0, cfg['syn']['w_max']
    )
    ax1.hist(w_init_samples, bins=50, alpha=0.7, density=True, 
             color='lightblue', label='Distribution initiale')
    ax1.set_xlabel('Poids synaptique')
    ax1.set_ylabel('Densité')
    ax1.set_title('Distribution des poids - Initial')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Poids finaux
    final_weights = np.array(monitors['syn_ee'].w)
    ax2.hist(final_weights, bins=50, alpha=0.7, density=True,
             color='orange', label='Distribution finale')
    ax2.set_xlabel('Poids synaptique')
    ax2.set_ylabel('Densité')
    ax2.set_title('Distribution des poids - Final')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'weight_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Trajectoires temporelles des poids échantillonnés
    if len(monitors['weights'].t) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        weight_times = monitors['weights'].t/1000  # en secondes
        weight_values = monitors['weights'].w
        
        
        # Afficher quelques trajectoires représentatives
        if weight_values.ndim > 1:
            n_show = min(10, weight_values.shape[1])
            indices_to_show = np.linspace(0, weight_values.shape[1]-1, n_show, dtype=int)
            
            for i in indices_to_show:
                if len(weight_times) == len(weight_values[:, i]):
                    ax.plot(weight_times, weight_values[:, i], alpha=0.7, linewidth=1)
            
            title_text = f'Évolution temporelle des poids E→E (échantillon de {n_show})'
        else:
            # Cas d'un seul poids échantillonné
            if len(weight_times) == len(weight_values):
                ax.plot(weight_times, weight_values, alpha=0.7, linewidth=1)
            title_text = 'Évolution temporelle des poids E→E (1 échantillon)'
        
        ax.set_xlabel('Temps (s)')
        ax.set_ylabel('Poids synaptique')
        ax.set_title(title_text)
        ax.grid(True, alpha=0.3)
        
        # Ligne de référence w_max
        ax.axhline(y=cfg['syn']['w_max'], color='red', linestyle='--', 
                  alpha=0.5, label=f"w_max = {cfg['syn']['w_max']}")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'weight_trajectories.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    log_info(f"Graphiques sauvegardés dans {fig_dir}")


def main():
    """Fonction principale."""
    args = parse_args()
    
    try:
        # Chargement de la configuration
        log_info(f"Chargement de la configuration: {args.config}")
        cfg = load_config(args.config)
        
        # Override des paramètres via CLI
        if args.duration_ms is not None:
            cfg['duration_ms'] = args.duration_ms
            log_info(f"Durée surchargée: {args.duration_ms} ms")
            
        if args.seed is not None:
            cfg['seed'] = args.seed
            log_info(f"Graine surchargée: {args.seed}")
        
        # Configuration des graines aléatoires
        set_seeds(cfg['seed'])
        
        # Création du répertoire de sortie
        if args.outdir:
            outdir = args.outdir
        else:
            outdir = create_output_dir()
        ensure_dir(outdir)
        log_info(f"Répertoire de sortie: {outdir}")
        
        # Sauvegarde de la configuration effective
        config_path = os.path.join(outdir, 'config_effective.json')
        save_json(cfg, config_path)
        
        # Construction du réseau
        net, monitors = build_network(cfg)
        
        # Simulation
        run_sim(net, cfg['duration_ms'], cfg['scaling'], monitors)
        
        # Calcul et affichage des statistiques
        stats = get_network_stats(monitors)
        log_info("=== STATISTIQUES DU RÉSEAU ===")
        log_info(f"Spikes totaux E: {stats['total_spikes_e']}")
        log_info(f"Spikes totaux I: {stats['total_spikes_i']}")
        log_info(f"Taux moyen E: {stats['mean_rate_e']:.2f} Hz")
        log_info(f"Taux moyen I: {stats['mean_rate_i']:.2f} Hz")
        log_info(f"Poids E→E - Moyenne: {stats['weight_mean']:.3f}")
        log_info(f"Poids E→E - Écart-type: {stats['weight_std']:.3f}")
        log_info(f"Poids E→E - Min/Max: {stats['weight_min']:.3f}/{stats['weight_max']:.3f}")
        
        if stats['has_nan_weights']:
            log_error("ATTENTION: Présence de NaN dans les poids!")
            return 1
        
        # Sauvegarde des résultats
        save_results(monitors, cfg, outdir)
        
        # Sauvegarde des statistiques
        stats_path = os.path.join(outdir, 'network_stats.json')
        save_json(stats, stats_path)
        
        # Génération des graphiques
        if args.plot:
            create_plots(monitors, cfg, outdir)
        
        log_info("=== SIMULATION TERMINÉE AVEC SUCCÈS ===")
        log_info(f"Résultats disponibles dans: {outdir}")
        
        return 0
        
    except Exception as e:
        log_error(f"Erreur lors de l'exécution: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main()) 