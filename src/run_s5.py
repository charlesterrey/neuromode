"""
Script d'exécution du modèle S-5: analyse de connectivité EEG-like (PDC/dDTF) + viz 3D.

Usage:
    python -m src.run_s5 --config configs/s5.json --plot
"""

import argparse
import os
import sys
import json
import numpy as np

from .utils.io import log_info, log_error, create_output_dir, save_json, ensure_dir


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Modèle S-5: Connectivité directionnelle EEG-like (PDC/dDTF) + viz 3D"
    )
    
    parser.add_argument('--config', type=str, default='configs/s5.json',
                       help='Chemin vers le fichier de configuration JSON')
    parser.add_argument('--spikes_file', type=str,
                       help='Fichier spikes existant (NPZ/CSV)')
    parser.add_argument('--groups', type=int, default=20,
                       help='Nombre de groupes/canaux LFP')
    parser.add_argument('--coords_csv', type=str,
                       help='Fichier CSV des coordonnées 3D')
    parser.add_argument('--outdir', type=str,
                       help='Répertoire de sortie')
    parser.add_argument('--plot', action='store_true',
                       help='Générer les graphiques')
    parser.add_argument('--seed', type=int,
                       help='Graine aléatoire')
    
    return parser.parse_args()


def load_spikes_from_s4(output_dir: str = "outputs") -> tuple:
    """
    Charge les spikes depuis une simulation S-4 précédente.
    
    Returns:
        spike_times_ms, spike_indices, n_neurons
    """
    # Recherche du fichier de spikes le plus récent
    spike_files = []
    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('spikes_e.csv'):
                    spike_files.append(os.path.join(root, file))
    
    if not spike_files:
        raise FileNotFoundError(f"Aucun fichier spikes trouvé dans {output_dir}")
    
    # Prendre le plus récent
    latest_file = max(spike_files, key=os.path.getmtime)
    log_info(f"Chargement spikes depuis: {latest_file}")
    
    # Chargement CSV
    import pandas as pd
    df = pd.read_csv(latest_file)
    
    spike_times_ms = df['time_ms'].values
    spike_indices = df['neuron_id'].values
    n_neurons = int(np.max(spike_indices)) + 1
    
    return spike_times_ms, spike_indices, n_neurons


def analyze_connectivity_s5(spike_times_ms, spike_indices, n_neurons, cfg):
    """Pipeline d'analyse de connectivité S-5."""
    from .signals.lfp import make_lfp_proxy, preprocess_for_var
    from .connectivity.var_pdc import compute_var_pipeline
    
    log_info("Construction signaux LFP proxy...")
    
    # Construction LFP multi-canaux
    lfp_signals, fs, groups = make_lfp_proxy(
        spike_times_ms, spike_indices, n_neurons,
        n_groups=cfg['groups'],
        dt_bin_ms=cfg['dt_bin_ms'],
        tau_ms=cfg['tau_alpha_ms'],
        method='alpha'
    )
    
    log_info(f"Signaux LFP: {lfp_signals.shape} @ {fs:.1f} Hz")
    
    # Préprocessing pour VAR
    lfp_processed = preprocess_for_var(lfp_signals, detrend=True)
    
    # Pipeline VAR + PDC/dDTF
    log_info("Ajustement modèle VAR + calcul PDC/dDTF...")
    
    var_cfg = cfg['var']
    p_opt = None if var_cfg['auto_select'] else var_cfg['p']
    
    results = compute_var_pipeline(
        lfp_processed, fs, cfg['bands'],
        p=p_opt, ridge_lambda=var_cfg['ridge_lambda']
    )
    
    log_info(f"Modèle VAR({results['p_opt']}) ajusté, stable: {results['stable']}")
    
    return results, groups


def save_connectivity_results(results, outdir):
    """Sauvegarde les résultats de connectivité."""
    
    # CSVs par bande
    for band_name, band_data in results['bands'].items():
        pdc_file = os.path.join(outdir, f'pdc_{band_name}.csv')
        ddtf_file = os.path.join(outdir, f'ddtf_{band_name}.csv')
        
        import pandas as pd
        pd.DataFrame(band_data['PDC']).to_csv(pdc_file, index=False)
        pd.DataFrame(band_data['dDTF']).to_csv(ddtf_file, index=False)
    
    # Métriques JSON
    metrics = {
        'p_opt': int(results['p_opt']),
        'stable': bool(results['stable']),
        'bands': {}
    }
    
    for band_name, band_data in results['bands'].items():
        pdc_mean = float(np.mean(band_data['PDC']))
        ddtf_mean = float(np.mean(band_data['dDTF']))
        
        metrics['bands'][band_name] = {
            'PDC_mean': pdc_mean,
            'dDTF_mean': ddtf_mean,
            'f_range': band_data['f_range']
        }
    
    metrics_file = os.path.join(outdir, 'metrics.json')
    save_json(metrics, metrics_file)
    
    log_info(f"Résultats connectivité sauvegardés dans {outdir}")


def create_simple_viz(results, outdir):
    """Création de visualisations simples."""
    import matplotlib.pyplot as plt
    
    fig_dir = os.path.join(outdir, 'figures')
    ensure_dir(fig_dir)
    
    # Heatmaps PDC/dDTF par bande
    for band_name, band_data in results['bands'].items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # PDC
        im1 = ax1.imshow(band_data['PDC'], cmap='viridis', vmin=0, vmax=1)
        ax1.set_title(f'PDC - {band_name}')
        ax1.set_xlabel('Canal source')
        ax1.set_ylabel('Canal cible')
        plt.colorbar(im1, ax=ax1)
        
        # dDTF
        im2 = ax2.imshow(band_data['dDTF'], cmap='plasma', vmin=0, vmax=1)
        ax2.set_title(f'dDTF - {band_name}')
        ax2.set_xlabel('Canal source')
        ax2.set_ylabel('Canal cible')
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'connectivity_{band_name}.png'), dpi=150)
        plt.close()
    
    log_info(f"Figures sauvegardées dans {fig_dir}")


def main():
    """Fonction principale."""
    args = parse_args()
    
    try:
        # Chargement configuration
        with open(args.config, 'r') as f:
            cfg = json.load(f)
        
        if args.seed is not None:
            cfg['seed'] = args.seed
        
        np.random.seed(cfg['seed'])
        
        # Création répertoire de sortie
        if args.outdir:
            outdir = args.outdir
        else:
            outdir = create_output_dir("outputs")
        ensure_dir(outdir)
        
        log_info(f"=== ANALYSE S-5: CONNECTIVITÉ EEG-LIKE ===")
        log_info(f"Répertoire de sortie: {outdir}")
        
        # Chargement des spikes
        if args.spikes_file:
            log_info(f"Chargement spikes depuis: {args.spikes_file}")
            # TODO: Implémenter chargement depuis fichier spécifique
            spike_times_ms, spike_indices, n_neurons = load_spikes_from_s4()
        else:
            spike_times_ms, spike_indices, n_neurons = load_spikes_from_s4()
        
        log_info(f"Spikes chargés: {len(spike_times_ms)} spikes, {n_neurons} neurones")
        
        # Analyse de connectivité
        results, groups = analyze_connectivity_s5(spike_times_ms, spike_indices, n_neurons, cfg)
        
        # Sauvegarde résultats
        save_connectivity_results(results, outdir)
        
        # Visualisations
        if args.plot:
            create_simple_viz(results, outdir)
        
        log_info("=== ANALYSE S-5 TERMINÉE AVEC SUCCÈS ===")
        log_info(f"Résultats disponibles dans: {outdir}")
        
        return 0
        
    except Exception as e:
        log_error(f"Erreur lors de l'exécution S-5: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main()) 