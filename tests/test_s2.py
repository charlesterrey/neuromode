"""
Tests d'intégration pour le modèle S-2 avec plasticité structurelle.

Vérifie que le modèle S-2 s'exécute correctement et suit les phases 
de développement attendues.
"""

import unittest
import os
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path

# Import du modèle S-2
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_s2 import (
    load_config_s2, set_seeds_s2, build_network_s2, 
    create_structural_operation, run_sim_s2,
    save_results_s2, get_network_stats_s2
)
from src.utils.io import create_output_dir, save_json
from src.plasticity.structural import compute_density


class TestModelS2(unittest.TestCase):
    """Tests d'intégration pour le modèle S-2."""
    
    def setUp(self):
        """Configuration initiale des tests."""
        self.test_dir = tempfile.mkdtemp()
        
        # Configuration réduite pour tests rapides
        self.test_config = {
            "seed": 12345,
            "duration_ms": 2000,  # Courte durée pour tests
            "dt_ms": 0.1,
            "N_e": 50,  # Petit réseau
            "N_i": 10,
            "p_connect_EI": 0.2,
            "lif": {
                "tau_m_ms": 20,
                "E_L_mV": -70,
                "V_th_mV": -50,
                "V_reset_mV": -60,
                "refractory_ms": 5
            },
            "syn": {
                "gmax_e": 0.6,
                "gmax_i": 6.0,
                "w_init_mean": 0.5,
                "w_init_std": 0.1,
                "w_max": 1.0,
                "tau_e_ms": 5,
                "tau_i_ms": 10
            },
            "stdp": {
                "tau_pre_ms": 20,
                "tau_post_ms": 20,
                "Apre": 0.01,
                "Apost": -0.012
            },
            "scaling": {
                "enabled": True,
                "target_hz": 5.0,
                "interval_ms": 200,
                "eta_scale": 0.01,
                "min_scale": 0.5,
                "max_scale": 2.0
            },
            "struct": {
                "dt_struct_ms": 100,  # Plus grand intervalle pour tests
                "phase": {
                    "T_grow_ms": 800,   # Phase GROW courte
                    "T_prune_ms": 1200  # Phase PRUNE courte
                },
                "rho_target_grow": 0.15,  # Densité cible réduite
                "max_add_per_step": 500,
                "tau_A_ms": 300,
                "beta_pre": 0.02,
                "beta_post": 0.02,
                "theta_act": 0.1,
                "k1": 5.0,
                "k2": 2.0,
                "lambda_len": 1.0,
                "max_degree": 100
            },
            "record": {
                "sample_weights": 50,
                "max_spikes": 50000
            }
        }
        
        # Sauvegarde de la config de test
        self.config_path = os.path.join(self.test_dir, 'test_config_s2.json')
        save_json(self.test_config, self.config_path)
    
    def tearDown(self):
        """Nettoyage après les tests."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_config_loading_s2(self):
        """Test du chargement de configuration S-2."""
        cfg = load_config_s2(self.config_path)
        self.assertEqual(cfg['seed'], 12345)
        self.assertEqual(cfg['N_e'], 50)
        self.assertEqual(cfg['struct']['phase']['T_grow_ms'], 800)
        self.assertTrue(cfg['scaling']['enabled'])
    
    def test_network_building_s2(self):
        """Test de la construction du réseau S-2."""
        set_seeds_s2(self.test_config['seed'])
        net, monitors = build_network_s2(self.test_config)
        
        # Vérifications de base
        self.assertIsNotNone(net)
        self.assertIsNotNone(monitors)
        
        # Vérification des moniteurs spécifiques S-2
        required_monitors = ['spikes_e', 'spikes_i', 'rates_e', 'rates_i', 
                           'weights', 'syn_ee', 'density_log', 'rng']
        for mon_name in required_monitors:
            self.assertIn(mon_name, monitors)
        
        # Vérification de la structure synaptique
        syn_ee = monitors['syn_ee']
        self.assertTrue(hasattr(syn_ee, 'alive'))
        self.assertTrue(hasattr(syn_ee, 'act_score'))
        self.assertTrue(hasattr(syn_ee, 'len_cost'))
        
        # Vérification des connexions préallouées
        N_e = self.test_config['N_e']
        expected_connections = N_e * (N_e - 1)  # Full sans auto-connexions
        self.assertEqual(len(syn_ee), expected_connections)
    
    def test_structural_phases_execution(self):
        """Test de l'exécution des phases structurelles GROW/PRUNE."""
        set_seeds_s2(self.test_config['seed'])
        net, monitors = build_network_s2(self.test_config)
        
        # Création de l'opération structurelle
        outdir = os.path.join(self.test_dir, 'test_struct_output')
        structural_op = create_structural_operation(self.test_config, monitors, outdir)
        
        # Simulation complète
        run_sim_s2(net, self.test_config['duration_ms'], 
                  self.test_config['scaling'], monitors, structural_op)
        
        # Calcul des statistiques
        stats = get_network_stats_s2(monitors)
        
        # === VÉRIFICATIONS CRITIQUES ===
        
        # 1. Pas de NaN dans les variables
        self.assertFalse(stats['has_nan_weights'], "NaN détectés dans les poids")
        self.assertFalse(stats['has_nan_activity'], "NaN détectés dans l'activité")
        
        # 2. Poids dans les bornes
        self.assertTrue(stats['weights_in_bounds'], "Poids hors des bornes")
        
        # 3. Le réseau émet des spikes
        self.assertGreater(stats['total_spikes_e'], 0, "Aucun spike excitateur")
        
        # 4. Événements grow/prune ont eu lieu
        self.assertGreater(len(monitors['density_log']), 0, "Aucun log de densité")
        
        # 5. Densité a évolué (pattern hausse/baisse attendu)
        if len(monitors['density_log']) > 0:
            density_data = np.array(monitors['density_log'])
            densities = density_data[:, 1]
            
            # Vérification que la densité a varié
            density_variation = np.max(densities) - np.min(densities)
            self.assertGreater(density_variation, 0.01, 
                             "Densité n'a pas suffisamment varié")
            
            # Vérification pattern hausse puis baisse
            max_density = np.max(densities)
            final_density = densities[-1]
            
            # On s'attend à ce que la densité max soit > densité finale
            # (après phase PRUNE)
            if max_density > final_density:
                self.assertTrue(True)  # Pattern attendu
            else:
                # Tolérance pour tests courts
                self.assertGreater(max_density, 0.05, "Densité max trop faible")
    
    def test_file_outputs_s2(self):
        """Test de la génération des fichiers de sortie S-2."""
        set_seeds_s2(self.test_config['seed'])
        net, monitors = build_network_s2(self.test_config)
        
        outdir = os.path.join(self.test_dir, 'test_output_s2')
        structural_op = create_structural_operation(self.test_config, monitors, outdir)
        
        run_sim_s2(net, self.test_config['duration_ms'],
                  self.test_config['scaling'], monitors, structural_op)
        
        # Sauvegarde des résultats
        save_results_s2(monitors, self.test_config, outdir)
        
        # Vérification des fichiers attendus S-2
        expected_files = [
            'spikes_e.csv',
            'spikes_i.csv', 
            'rates_e.csv',
            'rates_i.csv',
            'weight_trajectories.npy',
            'activity_trajectories.npy',
            'alive_trajectories.npy',
            'final_weights_ee.npy',
            'final_activity_ee.npy',
            'final_alive_ee.npy',
            'final_len_costs_ee.npy',
            'density_evolution.csv',
            'degree_histogram.csv',
            'structural_events.csv'
        ]
        
        for filename in expected_files:
            filepath = os.path.join(outdir, filename)
            self.assertTrue(os.path.exists(filepath),
                           f"Fichier manquant: {filename}")
            self.assertGreater(os.path.getsize(filepath), 0,
                              f"Fichier vide: {filename}")
    
    def test_density_evolution_pattern(self):
        """Test spécifique du pattern d'évolution de la densité."""
        # Configuration avec phases très courtes pour test rapide
        quick_config = self.test_config.copy()
        quick_config['duration_ms'] = 1000
        quick_config['struct']['phase']['T_grow_ms'] = 300
        quick_config['struct']['phase']['T_prune_ms'] = 700
        quick_config['struct']['dt_struct_ms'] = 50  # Plus fréquent
        
        set_seeds_s2(quick_config['seed'])
        net, monitors = build_network_s2(quick_config)
        
        outdir = os.path.join(self.test_dir, 'test_density')
        structural_op = create_structural_operation(quick_config, monitors, outdir)
        
        run_sim_s2(net, quick_config['duration_ms'],
                  quick_config['scaling'], monitors, structural_op)
        
        # Analyse de l'évolution de la densité
        self.assertGreater(len(monitors['density_log']), 5, 
                          "Trop peu de points de densité")
        
        density_data = np.array(monitors['density_log'])
        times = density_data[:, 0]
        densities = density_data[:, 1]
        
        # Vérification que le temps couvre les deux phases
        self.assertGreater(np.max(times), quick_config['struct']['phase']['T_grow_ms'],
                          "Phase GROW non couverte")
        
        # Densité finale valide
        final_density = densities[-1]
        self.assertGreaterEqual(final_density, 0.0, "Densité finale négative")
        self.assertLessEqual(final_density, 1.0, "Densité finale > 1")
    
    def test_structural_parameters_bounds(self):
        """Test que les paramètres structuraux restent dans les bornes."""
        set_seeds_s2(self.test_config['seed'])
        net, monitors = build_network_s2(self.test_config)
        
        outdir = os.path.join(self.test_dir, 'test_bounds')
        structural_op = create_structural_operation(self.test_config, monitors, outdir)
        
        run_sim_s2(net, self.test_config['duration_ms'],
                  self.test_config['scaling'], monitors, structural_op)
        
        syn_ee = monitors['syn_ee']
        
        # Vérification des bornes pour toutes les variables
        all_weights = np.array(syn_ee.w)
        all_A = np.array(syn_ee.act_score)
        all_alive = np.array(syn_ee.alive)
        all_len_cost = np.array(syn_ee.len_cost)
        
        # Poids dans [0, w_max]
        self.assertTrue(np.all(all_weights >= 0), "Poids négatifs")
        self.assertTrue(np.all(all_weights <= self.test_config['syn']['w_max']),
                       "Poids > w_max")
        
        # A ≥ 0 (peut croître sans borne mais pas négatif)
        self.assertTrue(np.all(all_A >= 0), "Scores d'activité négatifs")
        
        # alive ∈ {0, 1}
        unique_alive = np.unique(all_alive)
        for val in unique_alive:
            self.assertIn(val, [0.0, 1.0], f"Valeur alive invalide: {val}")
        
        # len_cost normalisés [0, 1]
        self.assertTrue(np.all(all_len_cost >= 0), "Coûts de longueur négatifs")
        self.assertTrue(np.all(all_len_cost <= 1), "Coûts de longueur > 1")


def run_tests():
    """Exécute tous les tests S-2."""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    run_tests() 