"""
Script d'exécution du modèle S-3 avec fenêtres critiques γ(t) et contraintes énergétiques.

Usage:
    python -m src.run_s3 --config configs/s3.json --plot
"""

import argparse
import os
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif

from .model_s3 import (
    load_config_s3, set_seeds_s3, build_network_s3, 
    create_structural_operation_s3, save_results_s3, get_network_stats_s3
)
from .model_s2 import run_sim_s2  # Réutilisation de la logique S-2
from .utils.io import log_info, log_error, create_output_dir, save_json, ensure_dir


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Modèle S-3: LIF + STDP + scaling + plasticité structurelle + γ(t) + énergie"
    )
    
    parser.add_argument('--config', type=str, default='configs/s3.json',
                       help='Chemin vers le fichier de configuration JSON')
    parser.add_argument('--outdir', type=str,
                       help='Répertoire de sortie (par défaut: outputs/run_TIMESTAMP)')
    parser.add_argument('--plot', action='store_true',
                       help='Générer les graphiques')
    parser.add_argument('--seed', type=int,
                       help='Graine aléatoire (override config)')
    
    return parser.parse_args()


def create_plots_s3(monitors, cfg, outdir):
    """Génère les graphiques spécialisés S-3 (γ, énergie, densité)."""
    log_info("Génération des graphiques S-3...")
    
    plt.style.use('default')
    fig_dir = os.path.join(outdir, 'figures')
    ensure_dir(fig_dir)
    
    # === 1. PROFIL γ(t) ===
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(monitors['gamma_log']) > 0:
        gamma_data = np.array(monitors['gamma_log'])
        times = gamma_data[:, 0] / 1000  # en secondes
        gammas = gamma_data[:, 1]
        
        ax.plot(times, gammas, linewidth=3, color='darkgreen', label='γ(t)')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='γ = 0.5')
        
        # Phases développementales
        T_grow = cfg['struct']['phase']['T_grow_ms'] / 1000
        ax.axvline(x=T_grow, color='red', linestyle='--', alpha=0.7, label='Fin GROW')
    
    ax.set_xlabel('Temps (s)')
    ax.set_ylabel('Facteur γ(t)')
    ax.set_title('Fenêtre critique γ(t) - Modulation développementale')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'gamma_profile.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # === 2. ÉNERGIE VS BUDGET ===
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    if len(monitors['energy_log']) > 0:
        energy_data = np.array(monitors['energy_log'])
        times = energy_data[:, 0] / 1000  # en secondes
        E_wins = energy_data[:, 1]
        P_Es = energy_data[:, 2]
        budgets = energy_data[:, 3]
        
        # Énergie vs budget
        ax1.plot(times, E_wins, linewidth=2, color='orange', label='Énergie E_win')
        ax1.axhline(y=budgets[0], color='red', linestyle='-', linewidth=2, alpha=0.8, label=f'Budget B = {budgets[0]:.0f}')
        ax1.fill_between(times, E_wins, budgets[0], where=(E_wins > budgets[0]), alpha=0.3, color='red', label='Dépassement')
        
        ax1.set_ylabel('Énergie')
        ax1.set_title('Budget énergétique et consommation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Pression énergétique
        ax2.plot(times, P_Es, linewidth=2, color='darkred', label='Pression P_E')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.fill_between(times, P_Es, 0, where=(P_Es > 0), alpha=0.3, color='red')
        
        ax2.set_xlabel('Temps (s)')
        ax2.set_ylabel('Pression P_E')
        ax2.set_title('Pression énergétique (élagage accéléré si P_E > 0)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'energy_budget.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # === 3. DENSITÉ AVEC γ(t) ET P_E ===
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Densité synaptique
    if len(monitors['density_log']) > 0:
        density_data = np.array(monitors['density_log'])
        times_d = density_data[:, 0] / 1000
        densities = density_data[:, 1]
        
        ax1.plot(times_d, densities, linewidth=2, color='darkblue', label='Densité')
        T_grow = cfg['struct']['phase']['T_grow_ms'] / 1000
        ax1.axvline(x=T_grow, color='red', linestyle='--', alpha=0.7)
        ax1.set_ylabel('Densité')
        ax1.set_title('Évolution densité avec modulation γ(t) et pression énergétique')
        ax1.grid(True, alpha=0.3)
    
    # γ(t) 
    if len(monitors['gamma_log']) > 0:
        gamma_data = np.array(monitors['gamma_log'])
        times_g = gamma_data[:, 0] / 1000
        gammas = gamma_data[:, 1]
        
        ax2.plot(times_g, gammas, linewidth=2, color='darkgreen', label='γ(t)')
        ax2.set_ylabel('γ(t)')
        ax2.grid(True, alpha=0.3)
    
    # Pression énergétique
    if len(monitors['energy_log']) > 0:
        energy_data = np.array(monitors['energy_log'])
        times_e = energy_data[:, 0] / 1000
        P_Es = energy_data[:, 2]
        
        ax3.plot(times_e, P_Es, linewidth=2, color='darkred', label='P_E')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_xlabel('Temps (s)')
        ax3.set_ylabel('P_E')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'density_gamma_energy.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    log_info(f"Graphiques S-3 sauvegardés dans {fig_dir}")


def main():
    """Fonction principale."""
    args = parse_args()
    
    try:
        # Chargement configuration S-3
        log_info(f"Chargement de la configuration S-3: {args.config}")
        cfg = load_config_s3(args.config)
        
        if args.seed is not None:
            cfg['seed'] = args.seed
            log_info(f"Graine surchargée: {args.seed}")
        
        set_seeds_s3(cfg['seed'])
        
        # Création répertoire de sortie
        if args.outdir:
            outdir = args.outdir
        else:
            outdir = create_output_dir("outputs")
        ensure_dir(outdir)
        log_info(f"Répertoire de sortie: {outdir}")
        
        # Sauvegarde config effective
        config_path = os.path.join(outdir, 'config_effective.json')
        save_json(cfg, config_path)
        
        # Construction réseau S-3
        net, monitors = build_network_s3(cfg)
        
        # Création opération structurelle S-3
        structural_op = create_structural_operation_s3(cfg, monitors, outdir)
        
        # Simulation (réutilisation logique S-2)
        run_sim_s2(net, cfg['duration_ms'], cfg['scaling'], monitors, structural_op)
        
        # Statistiques
        stats = get_network_stats_s3(monitors)
        log_info("=== STATISTIQUES DU RÉSEAU S-3 ===")
        log_info(f"Spikes totaux E: {stats['total_spikes_e']}")
        log_info(f"Spikes totaux I: {stats['total_spikes_i']}")
        log_info(f"Taux moyen E: {stats['mean_rate_e']:.2f} Hz")
        log_info(f"Taux moyen I: {stats['mean_rate_i']:.2f} Hz")
        log_info(f"Densité finale: {stats['final_density']:.4f}")
        log_info(f"γ moyen: {stats['gamma_mean']:.3f}, γ final: {stats['gamma_final']:.3f}")
        log_info(f"Énergie moyenne: {stats['energy_mean']:.1f}")
        log_info(f"Pression max: {stats['pressure_max']:.3f}")
        log_info(f"Budget dépassé: {stats['budget_exceeded_fraction']*100:.1f}% du temps")
        
        # Sauvegarde résultats S-3
        save_results_s3(monitors, cfg, outdir)
        
        # Statistiques
        stats_path = os.path.join(outdir, 'network_stats.json')
        save_json(stats, stats_path)
        
        # Graphiques
        if args.plot:
            create_plots_s3(monitors, cfg, outdir)
        
        log_info("=== SIMULATION S-3 TERMINÉE AVEC SUCCÈS ===")
        log_info(f"Résultats disponibles dans: {outdir}")
        
        return 0
        
    except Exception as e:
        log_error(f"Erreur lors de l'exécution S-3: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main()) 