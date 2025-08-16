"""
Script principal d'exécution de l'étude d'ablation S-6.

Usage:
    python -m src.run_s6 --config configs/s6.json --plot
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

from .utils.io import log_info, log_error, create_output_dir, save_json, ensure_dir
from .experiments.ablation_s6 import GridRunner


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Étude d'ablation S-6: grille Ω×t0×seeds"
    )
    
    parser.add_argument('--config', type=str, default='configs/s6.json',
                       help='Chemin vers le fichier de configuration JSON')
    parser.add_argument('--outdir', type=str,
                       help='Répertoire de sortie')
    parser.add_argument('--mode', type=str, choices=['run', 'replay'], 
                       default='run', help='Mode d\'exécution')
    parser.add_argument('--use_subprocess', action='store_true',
                       help='Utiliser subprocess au lieu d\'imports directs')
    parser.add_argument('--plot', action='store_true',
                       help='Générer les graphiques')
    parser.add_argument('--seed', type=int,
                       help='Graine aléatoire globale')
    
    return parser.parse_args()


def run_statistical_analysis(df: pd.DataFrame, outdir: str, config: dict):
    """
    Effectue les analyses statistiques sur les résultats.
    
    Args:
        df: DataFrame des résultats
        outdir: Répertoire de sortie
        config: Configuration S-6
    """
    from .stats.stats import anova_two_way, perm_test_diff, corr_spearman
    
    log_info("Analyses statistiques...")
    
    # Filtrage des données réussies
    successful = df[df['status'] == 'success'].copy()
    
    if len(successful) < 4:
        log_error("Pas assez de données pour analyses statistiques")
        return
    
    stats_results = {}
    
    try:
        # ANOVA bidirectionnelle: ODI ~ Omega * t0
        f_omega, p_omega, f_t0, p_t0, f_inter, p_inter = anova_two_way(
            successful, 'ODI', 'omega', 't0_ms'
        )
        
        stats_results['anova_odi'] = {
            'F_omega': float(f_omega), 'p_omega': float(p_omega),
            'F_t0': float(f_t0), 'p_t0': float(p_t0),
            'F_interaction': float(f_inter), 'p_interaction': float(p_inter)
        }
        
        # Test de permutation: Omega 0.0 vs 0.75
        omega_low = successful[successful['omega'] == 0.0]['ODI'].values
        omega_high = successful[successful['omega'] == 0.75]['ODI'].values
        
        if len(omega_low) > 0 and len(omega_high) > 0:
            # Créer un DataFrame temporaire pour le test
            temp_df = pd.DataFrame({
                'omega_group': ['low'] * len(omega_low) + ['high'] * len(omega_high),
                'ODI': np.concatenate([omega_low, omega_high])
            })
            
            p_perm = perm_test_diff(temp_df, 'omega_group', 'ODI')
            stats_results['permutation_omega'] = {'p_value': float(p_perm)}
        
        # Corrélations
        rho_omega, p_rho_omega = corr_spearman(successful, 'omega', 'ODI')
        rho_energy, p_rho_energy = corr_spearman(successful, 'E_total', 'ODI')
        
        stats_results['correlations'] = {
            'omega_odi': {'rho': float(rho_omega), 'p': float(p_rho_omega)},
            'energy_odi': {'rho': float(rho_energy), 'p': float(p_rho_energy)}
        }
        
    except Exception as e:
        log_error(f"Erreur analyses statistiques: {e}")
        stats_results['error'] = str(e)
    
    # Sauvegarde
    stats_file = os.path.join(outdir, 'statistical_analysis.json')
    save_json(stats_results, stats_file)
    
    log_info(f"Analyses statistiques sauvegardées: {stats_file}")


def create_report_and_figures(df: pd.DataFrame, outdir: str, config: dict):
    """
    Génère les tables publication-ready et figures.
    
    Args:
        df: DataFrame des résultats
        outdir: Répertoire de sortie
        config: Configuration S-6
    """
    from .report.report_s6 import make_tables, plots, build_html_report
    
    log_info("Génération des tables et figures...")
    
    try:
        # Tables publication-ready
        make_tables(df, outdir)
        
        # Figures
        plots(df, outdir)
        
        # Rapport HTML
        extra_paths = {
            'config': os.path.join(outdir, 'grid_config.json'),
            'stats': os.path.join(outdir, 'statistical_analysis.json')
        }
        build_html_report(df, outdir, extra_paths)
        
        log_info("Tables et figures générées avec succès")
        
    except Exception as e:
        log_error(f"Erreur génération rapport: {e}")


def print_summary(df: pd.DataFrame, runner: GridRunner):
    """
    Affiche un résumé final des résultats.
    
    Args:
        df: DataFrame des résultats
        runner: Instance GridRunner
    """
    successful = df[df['status'] == 'success']
    
    if len(successful) == 0:
        log_error("AUCUN RUN RÉUSSI - IMPOSSIBLE DE GÉNÉRER UN RÉSUMÉ")
        return
    
    print("\n" + "="*60)
    print("📊 RÉSUMÉ FINAL S-6")
    print("="*60)
    
    # Statistiques générales
    stats = runner.get_summary_stats(df)
    print(f"✅ Runs réussis: {stats['n_success']}/{stats['n_total']} ({stats['success_rate']:.1%})")
    
    # ODI par Omega
    print(f"\n📈 ODI MOYEN PAR OMEGA:")
    for omega, odi_mean in stats['ODI_by_omega'].items():
        print(f"   Ω={omega}: ODI={odi_mean:.3f}")
    
    # PDI par Omega
    print(f"\n🔗 PDI MOYEN PAR OMEGA:")
    for omega, pdi_mean in stats['PDI_by_omega'].items():
        print(f"   Ω={omega}: PDI={pdi_mean:.3f}")
    
    # Énergie
    print(f"\n⚡ ÉNERGIE:")
    print(f"   E_total moyen: {stats['E_total_mean']:.0f}")
    print(f"   E_overshoot moyen: {stats['E_overshoot_mean']:.1f}")
    
    # Plages de valeurs
    print(f"\n📊 PLAGES:")
    print(f"   ODI: [{stats['ODI_range'][0]:.3f}, {stats['ODI_range'][1]:.3f}]")
    print(f"   ODI σ: {stats['ODI_std']:.3f}")
    
    print("\n" + "="*60)


def main():
    """Fonction principale."""
    args = parse_args()
    
    try:
        # Chargement de la configuration
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        if args.seed is not None:
            config['global_seed'] = args.seed
            np.random.seed(args.seed)
        
        # Création du répertoire de sortie
        if args.outdir:
            outdir = args.outdir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            outdir = create_output_dir(f"outputs/s6_ablation_{timestamp}")
        ensure_dir(outdir)
        
        log_info(f"=== ÉTUDE D'ABLATION S-6 ===")
        log_info(f"Configuration: {args.config}")
        log_info(f"Répertoire de sortie: {outdir}")
        log_info(f"Mode: {args.mode}")
        
        # Initialisation du runner
        runner = GridRunner(config)
        
        # Exécution de la grille
        df = runner.run_grid(outdir, mode=args.mode, use_subprocess=args.use_subprocess)
        
        # Analyses statistiques
        run_statistical_analysis(df, outdir, config)
        
        # Génération des rapports
        if args.plot:
            create_report_and_figures(df, outdir, config)
        
        # Résumé final
        print_summary(df, runner)
        
        log_info("=== ÉTUDE S-6 TERMINÉE AVEC SUCCÈS ===")
        log_info(f"Résultats disponibles dans: {outdir}")
        
        return 0
        
    except Exception as e:
        log_error(f"Erreur lors de l'exécution S-6: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main()) 