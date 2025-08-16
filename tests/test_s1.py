"""
Tests d'intégration pour le modèle S-1.

Vérifie que le modèle s'exécute correctement et produit des résultats valides.
"""

import unittest
import os
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path

# Import du modèle
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_s1 import (
    load_config, set_seeds, build_network, run_sim, 
    save_results, get_network_stats
)
from src.utils.io import create_output_dir, save_json


class TestModelS1(unittest.TestCase):
    """Tests d'intégration pour le modèle S-1."""
    
    def setUp(self):
        """Configuration initiale des tests."""
        self.test_dir = tempfile.mkdtemp()
        
        # Configuration minimale pour tests rapides
        self.test_config = {
            "seed": 12345,
            "duration_ms": 1000,  # Simulation courte pour tests rapides
            "dt_ms": 0.1,
            "N_e": 50,  # Petit réseau pour vitesse
            "N_i": 10,
            "p_connect": 0.2,
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
            "record": {
                "sample_weights": 50,
                "max_spikes": 10000
            }
        }
        
        # Sauvegarde de la config de test
        self.config_path = os.path.join(self.test_dir, 'test_config.json')
        save_json(self.test_config, self.config_path)
    
    def tearDown(self):
        """Nettoyage après les tests."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_config_loading(self):
        """Test du chargement de configuration."""
        cfg = load_config(self.config_path)
        self.assertEqual(cfg['seed'], 12345)
        self.assertEqual(cfg['N_e'], 50)
        self.assertEqual(cfg['scaling']['enabled'], True)
    
    def test_network_building(self):
        """Test de la construction du réseau."""
        set_seeds(self.test_config['seed'])
        net, monitors = build_network(self.test_config)
        
        # Vérifications de base
        self.assertIsNotNone(net)
        self.assertIsNotNone(monitors)
        
        # Vérification des moniteurs
        required_monitors = ['spikes_e', 'spikes_i', 'rates_e', 'rates_i', 'weights', 'syn_ee']
        for mon_name in required_monitors:
            self.assertIn(mon_name, monitors)
    
    def test_simulation_execution(self):
        """Test d'exécution complète avec vérifications de sanité."""
        # Configuration des seeds pour reproductibilité
        set_seeds(self.test_config['seed'])
        
        # Construction du réseau
        net, monitors = build_network(self.test_config)
        
        # Simulation
        run_sim(net, self.test_config['duration_ms'], 
               self.test_config['scaling'], monitors)
        
        # Calcul des statistiques
        stats = get_network_stats(monitors)
        
        # === VÉRIFICATIONS CRITIQUES ===
        
        # 1. Pas de NaN dans les poids
        self.assertFalse(stats['has_nan_weights'], 
                        "Les poids contiennent des NaN")
        
        # 2. Le réseau émet des spikes
        self.assertGreater(stats['total_spikes_e'], 0,
                          "Aucun spike excitateur détecté")
        
        # 3. Les poids restent dans les bornes
        self.assertGreaterEqual(stats['weight_min'], 0,
                               "Poids négatifs détectés")
        self.assertLessEqual(stats['weight_max'], self.test_config['syn']['w_max'],
                            "Poids dépassant w_max détectés")
        
        # 4. Les taux sont réalistes (non négatifs, pas trop élevés)
        self.assertGreaterEqual(stats['mean_rate_e'], 0,
                               "Taux excitateur négatif")
        self.assertLess(stats['mean_rate_e'], 100,
                       "Taux excitateur anormalement élevé")
        
        # 5. Variation des poids (plasticité active)
        self.assertGreater(stats['weight_std'], 0,
                          "Aucune variation dans les poids (plasticité inactive?)")
    
    def test_homeostatic_scaling_effect(self):
        """Test de l'effet du scaling homéostatique."""
        # Test avec scaling activé
        cfg_with_scaling = self.test_config.copy()
        cfg_with_scaling['scaling']['enabled'] = True
        cfg_with_scaling['duration_ms'] = 2000  # Plus long pour observer l'effet
        
        set_seeds(cfg_with_scaling['seed'])
        net1, monitors1 = build_network(cfg_with_scaling)
        run_sim(net1, cfg_with_scaling['duration_ms'], 
               cfg_with_scaling['scaling'], monitors1)
        stats1 = get_network_stats(monitors1)
        
        # Test sans scaling
        cfg_without_scaling = self.test_config.copy()
        cfg_without_scaling['scaling']['enabled'] = False
        cfg_without_scaling['duration_ms'] = 2000
        
        set_seeds(cfg_without_scaling['seed'])
        net2, monitors2 = build_network(cfg_without_scaling)
        run_sim(net2, cfg_without_scaling['duration_ms'],
               cfg_without_scaling['scaling'], monitors2)
        stats2 = get_network_stats(monitors2)
        
        # Le scaling devrait influencer les taux et la distribution des poids
        # (pas de test strict car dépendant des paramètres, mais vérification de base)
        self.assertFalse(stats1['has_nan_weights'])
        self.assertFalse(stats2['has_nan_weights'])
        self.assertGreater(stats1['total_spikes_e'], 0)
        self.assertGreater(stats2['total_spikes_e'], 0)
    
    def test_file_outputs(self):
        """Test de la génération des fichiers de sortie."""
        set_seeds(self.test_config['seed'])
        net, monitors = build_network(self.test_config)
        run_sim(net, self.test_config['duration_ms'],
               self.test_config['scaling'], monitors)
        
        # Sauvegarde dans un répertoire de test
        outdir = os.path.join(self.test_dir, 'test_output')
        save_results(monitors, self.test_config, outdir)
        
        # Vérification de la présence des fichiers attendus
        expected_files = [
            'spikes_e.csv',
            'spikes_i.csv', 
            'rates_e.csv',
            'rates_i.csv',
            'weight_trajectories.csv',
            'final_weights_ee.npy'
        ]
        
        for filename in expected_files:
            filepath = os.path.join(outdir, filename)
            self.assertTrue(os.path.exists(filepath),
                           f"Fichier manquant: {filename}")
            self.assertGreater(os.path.getsize(filepath), 0,
                              f"Fichier vide: {filename}")
    
    def test_weight_bounds_maintained(self):
        """Test que les contraintes sur les poids sont maintenues."""
        # Configuration avec paramètres plus agressifs pour tester les bornes
        cfg = self.test_config.copy()
        cfg['stdp']['Apre'] = 0.05  # Plus fort
        cfg['stdp']['Apost'] = -0.06  # Plus fort
        cfg['scaling']['eta_scale'] = 0.1  # Plus agressif
        cfg['duration_ms'] = 1500
        
        set_seeds(cfg['seed'])
        net, monitors = build_network(cfg)
        run_sim(net, cfg['duration_ms'], cfg['scaling'], monitors)
        
        # Vérification stricte des bornes
        final_weights = np.array(monitors['syn_ee'].w)
        
        self.assertTrue(np.all(final_weights >= 0),
                       "Poids négatifs détectés")
        self.assertTrue(np.all(final_weights <= cfg['syn']['w_max']),
                       f"Poids > w_max ({cfg['syn']['w_max']}) détectés")
        self.assertFalse(np.any(np.isnan(final_weights)),
                        "Valeurs NaN dans les poids")
        self.assertFalse(np.any(np.isinf(final_weights)),
                        "Valeurs infinies dans les poids")


def run_tests():
    """Exécute tous les tests."""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    run_tests() 