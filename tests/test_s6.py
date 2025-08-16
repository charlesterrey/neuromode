"""
Tests de sanité pour le modèle S-6 (études d'ablation).
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import pandas as pd

# Ajout du chemin pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.ablation_s6 import GridRunner
from src.metrics.behavior import compute_ODI, bootstrap_ci
from src.metrics.structure import compute_PDI, adjacency_from_alive
from src.metrics.energy import energy_aggregates
from src.stats.stats import anova_two_way, perm_test_diff


class TestModelS6(unittest.TestCase):
    """Tests pour le modèle S-6."""
    
    def setUp(self):
        """Configuration des tests."""
        np.random.seed(42)
        
        # Configuration minimale pour tests
        self.config = {
            'omega': [0.0, 0.5],
            't0_ms': [2000],
            'seeds': [1],
            'replications': 1,
            's4_base_config': 'configs/s4.json'
        }
        
    def test_grid_runner_initialization(self):
        """Test initialisation GridRunner."""
        runner = GridRunner(self.config)
        
        self.assertEqual(len(runner.omega_values), 2)
        self.assertEqual(len(runner.t0_values), 1)
        self.assertEqual(len(runner.seeds), 1)
        self.assertEqual(runner.get_total_conditions(), 2)
    
    def test_conditions_generation(self):
        """Test génération des conditions."""
        runner = GridRunner(self.config)
        conditions = runner.generate_conditions()
        
        self.assertEqual(len(conditions), 2)
        
        # Vérifier structure des conditions
        for cond in conditions:
            self.assertIn('omega', cond)
            self.assertIn('t0_ms', cond)
            self.assertIn('seed', cond)
            self.assertIn('rep', cond)
            self.assertIn('condition_id', cond)
    
    def test_simulated_metrics(self):
        """Test génération de métriques simulées."""
        runner = GridRunner(self.config)
        
        condition = {
            'omega': 0.5,
            't0_ms': 2000,
            'seed': 1,
            'rep': 1
        }
        
        metrics = runner._simulate_metrics(condition)
        
        # Vérifier présence des métriques
        expected_metrics = ['ODI', 'PDI', 'E_total', 'E_overshoot', 
                          'densite_finale', 'FR_mean_E']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)
            self.assertFalse(np.isnan(metrics[metric]))
        
        # Vérifier cohérence des valeurs
        self.assertGreaterEqual(metrics['PDI'], 0.0)
        self.assertLessEqual(metrics['PDI'], 1.0)
        self.assertGreater(metrics['E_total'], 0.0)
    
    def test_behavioral_metrics(self):
        """Test des métriques comportementales."""
        # ODI
        odi = compute_ODI(0.8, 0.6)
        self.assertAlmostEqual(odi, 0.2, places=3)
        
        # Bootstrap CI
        x = np.array([0.1, 0.2, 0.15, 0.25, 0.18])
        mean_boot, ci_low, ci_high = bootstrap_ci(x, n_boot=100)
        
        self.assertAlmostEqual(mean_boot, np.mean(x), places=2)
        self.assertLessEqual(ci_low, mean_boot)
        self.assertGreaterEqual(ci_high, mean_boot)
    
    def test_structural_metrics(self):
        """Test des métriques structurelles."""
        # Matrices aléatoires
        np.random.seed(42)
        A_ref = np.random.randint(0, 2, (5, 5))
        A_cond = np.random.randint(0, 2, (5, 5))
        
        # PDI
        pdi = compute_PDI(A_ref, A_cond, 1000.0, 1200.0)
        
        self.assertIsInstance(pdi, float)
        self.assertGreaterEqual(pdi, 0.0)
        self.assertFalse(np.isnan(pdi))
    
    def test_statistical_functions(self):
        """Test des fonctions statistiques."""
        # Données test
        df_test = pd.DataFrame({
            'omega': [0.0, 0.0, 0.5, 0.5, 0.0, 0.5],
            't0_ms': [2000, 3000, 2000, 3000, 2000, 3000],
            'ODI': [0.1, 0.15, 0.3, 0.35, 0.12, 0.32]
        })
        
        # ANOVA
        f1, p1, f2, p2, fi, pi = anova_two_way(df_test, 'ODI', 'omega', 't0_ms')
        
        self.assertIsInstance(f1, float)
        self.assertIsInstance(p1, float)
        self.assertGreaterEqual(p1, 0.0)
        self.assertLessEqual(p1, 1.0)
        
        # Test de permutation
        df_perm = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [0.1, 0.2, 0.8, 0.9]
        })
        
        p_perm = perm_test_diff(df_perm, 'group', 'value', n_perm=100)
        self.assertIsInstance(p_perm, float)
        self.assertGreaterEqual(p_perm, 0.0)
        self.assertLessEqual(p_perm, 1.0)
    
    def test_grid_execution_simulated(self):
        """Test exécution de grille avec métriques simulées."""
        runner = GridRunner(self.config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Exécution en mode direct (métriques simulées)
            df = runner.run_grid(tmpdir, mode='direct', use_subprocess=False)
            
            # Vérifications DataFrame
            self.assertEqual(len(df), 2)
            self.assertIn('omega', df.columns)
            self.assertIn('ODI', df.columns)
            self.assertIn('PDI', df.columns)
            self.assertIn('status', df.columns)
            
            # Vérifier pas de NaN
            numeric_cols = ['ODI', 'PDI', 'E_total']
            for col in numeric_cols:
                if col in df.columns:
                    self.assertFalse(df[col].isna().any())
            
            # Vérifier différences par omega
            if len(df) > 1:
                omega_groups = df.groupby('omega')['ODI'].mean()
                if len(omega_groups) > 1:
                    # ODI devrait être plus élevé pour Omega plus grand
                    self.assertGreaterEqual(omega_groups.loc[0.5], omega_groups.loc[0.0] - 0.2)
    
    def test_summary_stats(self):
        """Test des statistiques de résumé."""
        runner = GridRunner(self.config)
        
        # DataFrame simulé
        df = pd.DataFrame({
            'omega': [0.0, 0.0, 0.5, 0.5],
            't0_ms': [2000, 2000, 2000, 2000],
            'status': ['success'] * 4,
            'ODI': [0.1, 0.12, 0.3, 0.32],
            'PDI': [0.05, 0.06, 0.15, 0.16],
            'E_total': [50000, 51000, 60000, 61000]
        })
        
        stats = runner.get_summary_stats(df)
        
        # Vérifications
        self.assertEqual(stats['n_total'], 4)
        self.assertEqual(stats['n_success'], 4)
        self.assertEqual(stats['success_rate'], 1.0)
        
        self.assertIn('ODI_mean', stats)
        self.assertIn('ODI_by_omega', stats)
        self.assertIsInstance(stats['ODI_by_omega'], dict)
    
    def test_energy_aggregates_fallback(self):
        """Test agrégation énergie avec fallback."""
        # Test avec fichier inexistant
        agg = energy_aggregates('/nonexistent/file.csv')
        
        self.assertIn('E_total', agg)
        self.assertEqual(agg['E_total'], 0.0)
        self.assertFalse(np.isnan(agg['E_total']))


def run_integration_test():
    """Test d'intégration rapide."""
    print("🧪 Test d'intégration S-6...")
    
    config = {
        'omega': [0.0, 0.3],
        't0_ms': [2500],
        'seeds': [42],
        'replications': 1,
        's4_base_config': 'configs/s4.json'
    }
    
    runner = GridRunner(config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        df = runner.run_grid(tmpdir, mode='direct', use_subprocess=False)
        
        assert len(df) == 2, "Nombre de conditions incorrect"
        assert all(df['status'] == 'success'), "Certains runs ont échoué"
        assert 'ODI' in df.columns, "Colonne ODI manquante"
        assert not df['ODI'].isna().any(), "NaN dans ODI"
        
        stats = runner.get_summary_stats(df)
        assert stats['n_success'] == 2, "Nombre de succès incorrect"
        
        print("✅ Test d'intégration S-6 réussi!")
        print(f"   - Conditions: {stats['n_success']}/{stats['n_total']}")
        print(f"   - ODI moyen: {stats['ODI_mean']:.3f}")


if __name__ == '__main__':
    # Tests unitaires
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Test d'intégration
    print("\n" + "="*50)
    run_integration_test() 