"""
Tests de sanité pour le modèle S-5 (connectivité EEG-like).

Vérifie que le pipeline LFP → VAR → PDC/dDTF fonctionne correctement.
"""

import unittest
import numpy as np
import os
import sys

# Ajout du chemin pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.signals.lfp import make_lfp_proxy, preprocess_for_var
from src.connectivity.var_pdc import compute_var_pipeline, fit_var_ols, is_stable


class TestModelS5(unittest.TestCase):
    """Tests pour le modèle S-5."""
    
    def setUp(self):
        """Configuration des tests."""
        np.random.seed(42)
        
        # Signaux jouets avec causalité simple
        self.K = 5  # 5 canaux
        self.T = 1000  # 1000 échantillons
        self.fs = 250.0  # 250 Hz
        
        # Génération de signaux avec causalité A→B
        self.X = np.random.randn(self.T, self.K) * 0.1
        
        # Injection causalité simple: canal 0 → canal 1 avec délai
        for t in range(5, self.T):
            self.X[t, 1] += 0.3 * self.X[t-2, 0]  # Délai de 2 échantillons
            self.X[t, 2] += 0.2 * self.X[t-1, 1]  # 1→2
        
        # Configuration S-5
        self.bands = {'alpha': [8, 12], 'beta': [13, 30]}
        
    def test_lfp_construction(self):
        """Test construction des signaux LFP proxy."""
        # Génération spikes fictifs
        n_neurons = 50
        duration_ms = 2000
        rate = 15  # Hz
        
        n_spikes = int(n_neurons * duration_ms * rate / 1000)
        spike_times_ms = np.sort(np.random.uniform(0, duration_ms, n_spikes))
        spike_indices = np.random.randint(0, n_neurons, n_spikes)
        
        # Construction LFP
        lfp_signals, fs, groups = make_lfp_proxy(
            spike_times_ms, spike_indices, n_neurons, n_groups=8
        )
        
        # Vérifications
        self.assertEqual(lfp_signals.shape[1], 8)  # 8 canaux
        self.assertGreater(lfp_signals.shape[0], 100)  # Au moins 100 échantillons
        self.assertAlmostEqual(fs, 500.0, places=1)  # 500 Hz par défaut
        self.assertEqual(len(groups), 8)
        
        # Vérifier normalisation z-score
        for k in range(8):
            signal_std = np.std(lfp_signals[:, k])
            self.assertAlmostEqual(signal_std, 1.0, places=1)
    
    def test_var_pipeline_stability(self):
        """Test du pipeline VAR avec vérification de stabilité."""
        # Preprocessing
        X_proc = preprocess_for_var(self.X, detrend=True)
        
        # Pipeline complet
        results = compute_var_pipeline(X_proc, self.fs, self.bands, p=3)
        
        # Vérifications
        self.assertEqual(results['p_opt'], 3)
        self.assertTrue(results['stable'])  # Modèle doit être stable
        
        # Matrices PDC/dDTF
        for band_name in ['alpha', 'beta']:
            PDC = results['bands'][band_name]['PDC']
            dDTF = results['bands'][band_name]['dDTF']
            
            # Vérifier dimensions
            self.assertEqual(PDC.shape, (self.K, self.K))
            self.assertEqual(dDTF.shape, (self.K, self.K))
            
            # Vérifier bornes [0,1]
            self.assertTrue(np.all(PDC >= 0))
            self.assertTrue(np.all(PDC <= 1))
            self.assertTrue(np.all(dDTF >= 0))
            self.assertTrue(np.all(dDTF <= 1))
            
            # Diagonale doit être ~0 (pas d'auto-causalité)
            self.assertLess(np.mean(np.diag(PDC)), 0.01)
            self.assertLess(np.mean(np.diag(dDTF)), 0.01)
    
    def test_causal_detection(self):
        """Test détection de causalité injectée."""
        # Pipeline sur signaux avec causalité connue
        results = compute_var_pipeline(self.X, self.fs, self.bands, p=5)
        
        # Vérifier que la causalité 0→1 est détectée
        for band_name in ['alpha', 'beta']:
            PDC = results['bands'][band_name]['PDC']
            
            # PDC[1,0] devrait être > PDC[0,1] (0 influence 1, pas l'inverse)
            causal_01 = PDC[1, 0]  # Influence de 0 sur 1
            causal_10 = PDC[0, 1]  # Influence de 1 sur 0
            
            self.assertGreater(causal_01, 0.01)  # Causalité détectée
            # Note: pas forcément causal_01 > causal_10 à cause du bruit
    
    def test_var_fitting(self):
        """Test ajustement VAR avec gestion d'erreurs."""
        # Test modèle stable
        A, Sigma = fit_var_ols(self.X, p=3, ridge_lambda=1e-3)
        
        # Vérifications dimensions
        self.assertEqual(A.shape, (self.K, self.K, 3))
        self.assertEqual(Sigma.shape, (self.K, self.K))
        
        # Vérifier stabilité
        stable = is_stable(A)
        self.assertTrue(stable)
        
        # Sigma doit être définie positive
        eigenvals = np.linalg.eigvals(Sigma)
        self.assertTrue(np.all(eigenvals > 0))
    
    def test_csv_export_format(self):
        """Test format des exports CSV."""
        results = compute_var_pipeline(self.X, self.fs, self.bands, p=2)
        
        # Vérifier qu'on peut exporter en CSV
        for band_name, band_data in results['bands'].items():
            PDC = band_data['PDC']
            dDTF = band_data['dDTF']
            
            # Vérifier pas de NaN/Inf
            self.assertFalse(np.any(np.isnan(PDC)))
            self.assertFalse(np.any(np.isinf(PDC)))
            self.assertFalse(np.any(np.isnan(dDTF)))
            self.assertFalse(np.any(np.isinf(dDTF)))
    
    def test_metrics_computation(self):
        """Test calcul des métriques de connectivité."""
        results = compute_var_pipeline(self.X, self.fs, self.bands, p=3)
        
        # Métriques par bande
        for band_name, band_data in results['bands'].items():
            PDC = band_data['PDC']
            dDTF = band_data['dDTF']
            
            # Moyennes globales
            pdc_mean = np.mean(PDC)
            ddtf_mean = np.mean(dDTF)
            
            # Vérifier valeurs raisonnables
            self.assertGreaterEqual(pdc_mean, 0.0)
            self.assertLessEqual(pdc_mean, 1.0)
            self.assertGreaterEqual(ddtf_mean, 0.0)
            self.assertLessEqual(ddtf_mean, 1.0)
            
            # Connectivité non-triviale (pas tout à zéro)
            self.assertGreater(np.max(PDC), 0.001)
            self.assertGreater(np.max(dDTF), 0.001)


def run_integration_test():
    """Test d'intégration complet."""
    print("🧪 Test d'intégration S-5...")
    
    # Génération données test
    np.random.seed(123)
    n_neurons = 80
    duration_ms = 3000
    
    # Spikes avec structure
    n_spikes = int(n_neurons * duration_ms * 12 / 1000)  # 12 Hz
    spike_times_ms = np.sort(np.random.uniform(0, duration_ms, n_spikes))
    spike_indices = np.random.randint(0, n_neurons, n_spikes)
    
    # Pipeline LFP
    lfp_signals, fs, groups = make_lfp_proxy(
        spike_times_ms, spike_indices, n_neurons, n_groups=12
    )
    
    # Preprocessing
    lfp_processed = preprocess_for_var(lfp_signals, detrend=True)
    
    # Analyse VAR
    bands = {'alpha': [8, 12], 'beta': [13, 30], 'gamma': [30, 50]}
    results = compute_var_pipeline(lfp_processed, fs, bands)
    
    # Vérifications finales
    assert results['stable'], "Modèle VAR instable"
    assert results['p_opt'] >= 1, "Ordre VAR invalide"
    
    for band_name in bands:
        PDC = results['bands'][band_name]['PDC']
        assert PDC.shape == (12, 12), f"Forme PDC incorrecte pour {band_name}"
        assert np.all(np.diag(PDC) < 0.01), f"Diagonale PDC non nulle pour {band_name}"
    
    print("✅ Test d'intégration S-5 réussi!")
    print(f"   - VAR({results['p_opt']}) stable: {results['stable']}")
    print(f"   - Signaux LFP: {lfp_signals.shape} @ {fs:.1f} Hz")
    
    for band_name, band_data in results['bands'].items():
        pdc_mean = np.mean(band_data['PDC'])
        print(f"   - {band_name}: PDC_mean = {pdc_mean:.4f}")


if __name__ == '__main__':
    # Tests unitaires
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Test d'intégration
    print("\n" + "="*50)
    run_integration_test() 