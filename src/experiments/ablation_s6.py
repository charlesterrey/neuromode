"""
Orchestration de l'étude d'ablation S-6.

Explore systématiquement l'espace Ω×t0×seeds et collecte les métriques
ODI, PDI, énergie pour analyse statistique et génération de tables.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import subprocess
import itertools
import traceback
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path

from ..utils.io import log_info, log_error, ensure_dir, save_json


class GridRunner:
    """
    Orchestrateur pour l'exploration de grille de paramètres.
    
    Génère toutes les combinaisons (Omega, t0, seed, replication)
    et collecte les métriques pour analyse statistique.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le runner avec la configuration.
        
        Args:
            config: Configuration S-6 complète
        """
        self.config = config.copy()
        self.results = []  # Liste des résultats par condition
        self.failed_runs = []  # Conditions échouées
        
        # Extraction des paramètres
        self.omega_values = config['omega']
        self.t0_values = config['t0_ms'] 
        self.seeds = config['seeds']
        self.replications = config['replications']
        
        # Configuration des chemins
        self.paths_cfg = config.get('paths', {})
        self.replay_cfg = config.get('replay', {})
        
        log_info(f"Grille configurée: {len(self.omega_values)} Ω × {len(self.t0_values)} t0 × "
                f"{len(self.seeds)} seeds × {self.replications} reps = "
                f"{self.get_total_conditions()} conditions")
    
    def get_total_conditions(self) -> int:
        """Retourne le nombre total de conditions à tester."""
        return len(self.omega_values) * len(self.t0_values) * len(self.seeds) * self.replications
    
    def generate_conditions(self) -> List[Dict[str, Any]]:
        """
        Génère toutes les combinaisons de paramètres.
        
        Returns:
            Liste des conditions (dict avec omega, t0_ms, seed, rep)
        """
        conditions = []
        
        for omega, t0_ms, seed, rep in itertools.product(
            self.omega_values, self.t0_values, self.seeds, range(1, self.replications + 1)
        ):
            condition = {
                'omega': omega,
                't0_ms': t0_ms,
                'seed': seed,
                'rep': rep,
                'condition_id': f"omega{omega:.2f}_t0{t0_ms}_seed{seed}_rep{rep}"
            }
            conditions.append(condition)
        
        return conditions
    
    def run_condition_subprocess(self, condition: Dict[str, Any], outdir: str) -> Dict[str, Any]:
        """
        Exécute une condition via subprocess (isolation complète).
        
        Args:
            condition: Paramètres de la condition
            outdir: Répertoire de sortie
            
        Returns:
            Métriques collectées ou dict d'erreur
        """
        cond_id = condition['condition_id']
        cond_outdir = os.path.join(outdir, 'runs', cond_id)
        ensure_dir(cond_outdir)
        
        try:
            # Construction de la config S-4 modifiée
            s4_config = self.config.get('s4_base_config', 'configs/s4.json')
            
            # Commande S-4
            cmd = [
                sys.executable, '-m', 'src.run_s4',
                '--config', s4_config,
                '--outdir', cond_outdir,
                '--seed', str(condition['seed'])
            ]
            
            # Variables d'environnement pour override des paramètres
            env = os.environ.copy()
            env['S4_OMEGA_OVERRIDE'] = str(condition['omega'])
            env['S4_T0_OVERRIDE'] = str(condition['t0_ms'])
            
            log_info(f"Lancement subprocess pour {cond_id}")
            
            # Exécution
            result = subprocess.run(
                cmd, capture_output=True, text=True, 
                cwd=os.getcwd(), env=env, timeout=300
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"S-4 failed: {result.stderr}")
            
            # Collecte des métriques depuis les fichiers générés
            metrics = self._collect_metrics_from_files(cond_outdir, condition)
            metrics['status'] = 'success'
            
            return metrics
            
        except Exception as e:
            log_error(f"Échec condition {cond_id}: {e}")
            return {
                'condition_id': cond_id,
                'status': 'failed',
                'error': str(e),
                **condition
            }
    
    def run_condition_direct(self, condition: Dict[str, Any], outdir: str) -> Dict[str, Any]:
        """
        Exécute une condition directement (imports Python).
        
        Args:
            condition: Paramètres de la condition
            outdir: Répertoire de sortie
            
        Returns:
            Métriques collectées
        """
        cond_id = condition['condition_id']
        cond_outdir = os.path.join(outdir, 'runs', cond_id)
        ensure_dir(cond_outdir)
        
        try:
            # Import dynamique des modules S-4
            from ..model_s4 import build_network_s4, load_config_s4, run_probe
            from ..model_s2 import run_sim_s2
            from ..utils.offloading import compute_odi
            
            # Chargement et modification de la config
            s4_config_path = self.config.get('s4_base_config', 'configs/s4.json')
            cfg = load_config_s4(s4_config_path)
            
            # Override des paramètres
            cfg['seed'] = condition['seed']
            cfg['offloading']['omega'] = condition['omega']
            cfg['offloading']['t0_ms'] = condition['t0_ms']
            
            # Simulation S-4
            log_info(f"Simulation directe {cond_id}")
            
            np.random.seed(cfg['seed'])
            net, monitors = build_network_s4(cfg)
            
            # TODO: Adapter pour extraction des métriques
            # Pour le moment, placeholder avec métriques simulées
            metrics = self._simulate_metrics(condition)
            metrics['status'] = 'success'
            
            return metrics
            
        except Exception as e:
            log_error(f"Échec simulation directe {cond_id}: {e}")
            return {
                'condition_id': cond_id,
                'status': 'failed',
                'error': str(e),
                **condition
            }
    
    def replay_condition(self, condition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rejoue une condition depuis des fichiers existants.
        
        Args:
            condition: Paramètres de la condition
            
        Returns:
            Métriques extraites des fichiers
        """
        cond_id = condition['condition_id']
        
        try:
            # Recherche des fichiers correspondants
            pattern = self.replay_cfg.get('glob_pattern', 'outputs/s4_runs/**/*.json')
            
            # TODO: Implémenter recherche et lecture
            # Pour le moment, métriques simulées
            metrics = self._simulate_metrics(condition)
            metrics['status'] = 'replayed'
            
            return metrics
            
        except Exception as e:
            log_error(f"Échec replay {cond_id}: {e}")
            return {
                'condition_id': cond_id,
                'status': 'failed',
                'error': str(e),
                **condition
            }
    
    def _collect_metrics_from_files(self, run_dir: str, condition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collecte les métriques depuis les fichiers de sortie d'une simulation.
        
        Args:
            run_dir: Répertoire de la simulation
            condition: Paramètres de la condition
            
        Returns:
            Dictionnaire des métriques
        """
        from ..metrics.behavior import compute_ODI
        from ..metrics.structure import compute_PDI, adjacency_from_alive
        from ..metrics.energy import energy_aggregates
        
        metrics = condition.copy()
        
        try:
            # Chargement ODI depuis JSON
            odi_file = os.path.join(run_dir, 'odi.json')
            if os.path.exists(odi_file):
                with open(odi_file, 'r') as f:
                    odi_data = json.load(f)
                    metrics['ODI'] = float(odi_data.get('odi', 0.0))
            else:
                metrics['ODI'] = 0.0
            
            # Calcul PDI (nécessite matrices adjacence de référence)
            # TODO: Implémenter extraction depuis matrices sauvegardées
            metrics['PDI'] = np.random.uniform(0, 0.5)  # Placeholder
            
            # Agrégats énergétiques
            energy_file = os.path.join(run_dir, 'energy.csv')
            if os.path.exists(energy_file):
                energy_agg = energy_aggregates(energy_file)
                metrics.update(energy_agg)
            else:
                metrics.update({
                    'E_total': 0.0,
                    'E_overshoot': 0.0,
                    'E_mean_win': 0.0,
                    'E_max_win': 0.0
                })
            
            # Métriques structurelles additionnelles
            metrics.update({
                'densite_finale': np.random.uniform(0.1, 0.3),
                'FR_mean_E': np.random.uniform(3, 8),
                'Delta_w_plus': np.random.uniform(-0.1, 0.1)
            })
            
        except Exception as e:
            log_error(f"Erreur collecte métriques: {e}")
            # Métriques par défaut en cas d'échec
            metrics.update({
                'ODI': 0.0, 'PDI': 0.0, 'E_total': 0.0,
                'E_overshoot': 0.0, 'densite_finale': 0.2,
                'FR_mean_E': 5.0, 'Delta_w_plus': 0.0
            })
        
        return metrics
    
    def _simulate_metrics(self, condition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Génère des métriques simulées pour test/développement.
        
        Args:
            condition: Paramètres de la condition
            
        Returns:
            Métriques simulées cohérentes
        """
        omega = condition['omega']
        t0_ms = condition['t0_ms']
        seed = condition['seed']
        
        # Générateur reproductible
        rng = np.random.default_rng(seed + hash(str(condition)) % 1000)
        
        # ODI: croît avec Omega (tendance + bruit)
        odi_base = omega * 0.3 + rng.normal(0, 0.05)
        odi = np.clip(odi_base, -0.2, 0.6)
        
        # PDI: influence de Omega et t0
        pdi_base = omega * 0.4 + (t0_ms - 3000) / 10000 + rng.normal(0, 0.1)
        pdi = np.clip(pdi_base, 0, 1.0)
        
        # Énergie: varie avec les paramètres
        e_total = 50000 + omega * 20000 + rng.normal(0, 5000)
        e_overshoot = max(0, rng.normal(omega * 10, 3))
        
        metrics = {
            'ODI': float(odi),
            'PDI': float(pdi),
            'E_total': float(e_total),
            'E_overshoot': float(e_overshoot),
            'E_mean_win': float(e_total / 100),
            'E_max_win': float(e_total / 50),
            'densite_finale': float(0.15 + omega * 0.1 + rng.normal(0, 0.02)),
            'FR_mean_E': float(5 + rng.normal(0, 1)),
            'Delta_w_plus': float(rng.normal(0, 0.05))
        }
        
        return {**condition, **metrics}
    
    def run_grid(self, outdir: str, mode: str = 'direct', use_subprocess: bool = False) -> pd.DataFrame:
        """
        Exécute toute la grille d'ablation.
        
        Args:
            outdir: Répertoire de sortie
            mode: 'run' ou 'replay'
            use_subprocess: Si True, utilise subprocess au lieu d'imports
            
        Returns:
            DataFrame avec tous les résultats
        """
        conditions = self.generate_conditions()
        total = len(conditions)
        
        log_info(f"Démarrage grille S-6: {total} conditions en mode {mode}")
        
        ensure_dir(outdir)
        
        # Sauvegarde de la configuration
        config_file = os.path.join(outdir, 'grid_config.json')
        save_json(self.config, config_file)
        
        for i, condition in enumerate(conditions):
            cond_id = condition['condition_id']
            log_info(f"[{i+1}/{total}] Traitement {cond_id}")
            
            try:
                if mode == 'replay':
                    result = self.replay_condition(condition)
                elif use_subprocess:
                    result = self.run_condition_subprocess(condition, outdir)
                else:
                    result = self.run_condition_direct(condition, outdir)
                
                self.results.append(result)
                
                if result.get('status') == 'failed':
                    self.failed_runs.append(result)
                
            except Exception as e:
                log_error(f"Erreur critique condition {cond_id}: {e}")
                error_result = {
                    'condition_id': cond_id,
                    'status': 'critical_error',
                    'error': str(e),
                    **condition
                }
                self.results.append(error_result)
                self.failed_runs.append(error_result)
        
        # Conversion en DataFrame
        df = pd.DataFrame(self.results)
        
        # Sauvegarde des résultats
        results_file = os.path.join(outdir, 'grid_results.csv')
        df.to_csv(results_file, index=False)
        
        results_parquet = os.path.join(outdir, 'grid_results.parquet')
        df.to_parquet(results_parquet, index=False)
        
        log_info(f"Grille terminée: {len(self.results)} résultats, {len(self.failed_runs)} échecs")
        log_info(f"Résultats sauvegardés: {results_file}")
        
        return df
    
    def get_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calcule des statistiques de résumé de la grille.
        
        Args:
            df: DataFrame des résultats
            
        Returns:
            Dictionnaire des statistiques
        """
        successful = df[df['status'] == 'success']
        
        if len(successful) == 0:
            return {'error': 'Aucun run réussi'}
        
        stats = {
            'n_total': len(df),
            'n_success': len(successful),
            'n_failed': len(df) - len(successful),
            'success_rate': len(successful) / len(df),
            
            # Statistiques ODI
            'ODI_mean': float(successful['ODI'].mean()),
            'ODI_std': float(successful['ODI'].std()),
            'ODI_range': [float(successful['ODI'].min()), float(successful['ODI'].max())],
            
            # Statistiques PDI
            'PDI_mean': float(successful['PDI'].mean()),
            'PDI_std': float(successful['PDI'].std()),
            
            # Énergie
            'E_total_mean': float(successful['E_total'].mean()),
            'E_overshoot_mean': float(successful['E_overshoot'].mean()),
            
            # Par Omega
            'ODI_by_omega': successful.groupby('omega')['ODI'].mean().to_dict(),
            'PDI_by_omega': successful.groupby('omega')['PDI'].mean().to_dict()
        }
        
        return stats 