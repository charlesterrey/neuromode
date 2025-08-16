"""
Script d'exécution du modèle S-4 avec offloading Ω(t) et contrôle LC-NE.

Usage:
    python -m src.run_s4 --config configs/s4.json --plot
"""

import argparse
import os
import sys
import numpy as np
import json

from .model_s4 import (
    load_config_s4, set_seeds_s4, build_network_s4, 
    create_structural_operation_s4, save_results_s4, get_network_stats_s4,
    run_probe
)
from .model_s2 import run_sim_s2  # Réutilisation logique S-2
from .utils.io import log_info, log_error, create_output_dir, save_json, ensure_dir
from .utils.offloading import compute_odi


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Modèle S-4: LIF + STDP + scaling + γ(t) + énergie + offloading + LC-NE"
    )
    
    parser.add_argument('--config', type=str, default='configs/s4.json',
                       help='Chemin vers le fichier de configuration JSON')
    parser.add_argument('--outdir', type=str,
                       help='Répertoire de sortie (par défaut: outputs/run_TIMESTAMP)')
    parser.add_argument('--plot', action='store_true',
                       help='Générer les graphiques')
    parser.add_argument('--seed', type=int,
                       help='Graine aléatoire (override config)')
    
    return parser.parse_args()


def run_s4_simulation_with_probes(net, monitors, cfg, structural_op, outdir):
    """
    Lance la simulation S-4 en 3 phases + probes pour calculer ODI.
    
    Returns:
        dict: Résultats avec scores de rappel et ODI
    """
    duration_ms = cfg['duration_ms']
    t0_ms = cfg['offloading']['t0_ms']
    
    log_info(f"=== PHASE 1: PRÉ-OFFLOADING (0 à {t0_ms}ms) ===")
    log_info("Effort endogène maximal, g_NE élevé, plasticité forte")
    
    # Simulation complète avec opération structurelle
    run_sim_s2(net, duration_ms, cfg['scaling'], monitors, structural_op)
    
    log_info(f"=== PHASE 2: POST-OFFLOADING ({t0_ms} à {duration_ms}ms) ===")
    log_info("Effort réduit, g_NE diminué, plasticité atténuée")
    
    log_info("=== PHASE 3: PROBES DE RAPPEL ===")
    
    # Sonde sans assistance (effort endogène pur)
    log_info("Probe sans assistance...")
    recall_noassist = run_probe(net, monitors, cfg, assist_mode=False)
    
    # Sonde avec assistance (stimulation externe)
    log_info("Probe avec assistance...")
    recall_assist = run_probe(net, monitors, cfg, assist_mode=True)
    
    # Calcul ODI
    odi = compute_odi(recall_noassist, recall_assist)
    
    results = {
        'recall_noassist': float(recall_noassist),
        'recall_assist': float(recall_assist),
        'odi': float(odi),
        'interpretation': 'ODI>0 indique dépendance à l\'aide externe'
    }
    
    log_info(f"Rappel sans aide: {recall_noassist:.2f}")
    log_info(f"Rappel avec aide: {recall_assist:.2f}")
    log_info(f"ODI: {odi:.3f} ({'Dépendance' if odi > 0 else 'Autonomie'})")
    
    # Sauvegarde ODI
    odi_path = os.path.join(outdir, 'odi.json')
    save_json(results, odi_path)
    
    return results


def main():
    """Fonction principale."""
    args = parse_args()
    
    try:
        # Chargement configuration S-4
        log_info(f"Chargement de la configuration S-4: {args.config}")
        cfg = load_config_s4(args.config)
        
        if args.seed is not None:
            cfg['seed'] = args.seed
            log_info(f"Graine surchargée: {args.seed}")
        
        set_seeds_s4(cfg['seed'])
        
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
        
        # Construction réseau S-4
        net, monitors = build_network_s4(cfg)
        
        # Création opération structurelle S-4
        structural_op = create_structural_operation_s4(cfg, monitors, outdir)
        
        # Simulation avec probes ODI
        odi_results = run_s4_simulation_with_probes(net, monitors, cfg, structural_op, outdir)
        
        # Statistiques
        stats = get_network_stats_s4(monitors)
        log_info("=== STATISTIQUES DU RÉSEAU S-4 ===")
        log_info(f"Spikes totaux E: {stats['total_spikes_e']}")
        log_info(f"Taux moyen E: {stats['mean_rate_e']:.2f} Hz")
        log_info(f"Densité finale: {stats['final_density']:.4f}")
        log_info(f"γ moyen: {stats['gamma_mean']:.3f}")
        log_info(f"Effort moyen: {stats['effort_mean']:.3f}, final: {stats['effort_final']:.3f}")
        log_info(f"g_NE moyen: {stats['gne_mean']:.3f}, final: {stats['gne_final']:.3f}")
        log_info(f"ODI: {odi_results['odi']:.3f}")
        
        # Sauvegarde résultats S-4
        save_results_s4(monitors, cfg, outdir)
        
        # Statistiques avec ODI
        stats.update(odi_results)
        stats_path = os.path.join(outdir, 'network_stats.json')
        save_json(stats, stats_path)
        
        log_info("=== SIMULATION S-4 TERMINÉE AVEC SUCCÈS ===")
        log_info(f"Résultats disponibles dans: {outdir}")
        log_info(f"ODI = {odi_results['odi']:.3f} ({'dépendance' if odi_results['odi'] > 0 else 'autonomie'} à l'aide externe)")
        
        return 0
        
    except Exception as e:
        log_error(f"Erreur lors de l'exécution S-4: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main()) 