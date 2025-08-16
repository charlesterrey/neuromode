"""
Module de planification des fenêtres critiques γ(t).

Implémente différents profils temporels pour moduler l'apprentissage
et la plasticité selon les théories des périodes critiques développementales.
"""

import numpy as np
from typing import Dict, Any, Union


class GammaScheduler:
    """
    Planificateur de fenêtres critiques γ(t) pour modulation développementale.
    
    Supporte plusieurs modes de planification pour différents profils
    neurobiologiques de périodes critiques.
    """
    
    def __init__(self, mode: str, params: Dict[str, Any]):
        """
        Initialise le planificateur γ(t).
        
        Args:
            mode: Type de profil ('double_sigmoid', 'gauss')
            params: Paramètres spécifiques au mode choisi
        """
        self.mode = mode.lower()
        self.params = params.copy()
        
        # Validation des paramètres selon le mode
        if self.mode == 'double_sigmoid':
            required = ['gamma_max', 't_open_ms', 't_close_ms', 'slope_open', 'slope_close']
            for param in required:
                if param not in params:
                    raise ValueError(f"Paramètre manquant pour mode double_sigmoid: {param}")
                    
        elif self.mode == 'gauss':
            required = ['gamma_max', 't_peak_ms', 'sigma']
            for param in required:
                if param not in params:
                    raise ValueError(f"Paramètre manquant pour mode gauss: {param}")
                    
        else:
            raise ValueError(f"Mode γ(t) non supporté: {mode}")
    
    def value(self, t_ms: float) -> float:
        """
        Calcule la valeur de γ(t) au temps donné.
        
        Args:
            t_ms: Temps en millisecondes
            
        Returns:
            Valeur de γ(t) dans [0, gamma_max]
        """
        if self.mode == 'double_sigmoid':
            return self._double_sigmoid(t_ms)
        elif self.mode == 'gauss':
            return self._gaussian(t_ms)
        else:
            return 1.0  # Fallback
    
    def _double_sigmoid(self, t_ms: float) -> float:
        """
        Profil double-sigmoïde : ouverture puis fermeture de fenêtre critique.
        
        Formule: γ(t) = γ_max * σ(k_open*(t-t_open)) * σ(k_close*(t_close-t))
        
        Args:
            t_ms: Temps en millisecondes
            
        Returns:
            Valeur γ(t) pour fenêtre critique avec ouverture et fermeture
        """
        gamma_max = self.params['gamma_max']
        t_open = self.params['t_open_ms']
        t_close = self.params['t_close_ms']
        slope_open = self.params['slope_open']
        slope_close = self.params['slope_close']
        
        # Sigmoïde d'ouverture (augmentation)
        sigmoid_open = 1.0 / (1.0 + np.exp(-slope_open * (t_ms - t_open)))
        
        # Sigmoïde de fermeture (diminution)
        sigmoid_close = 1.0 / (1.0 + np.exp(-slope_close * (t_close - t_ms)))
        
        # Produit des deux sigmoïdes pour créer une fenêtre
        gamma = gamma_max * sigmoid_open * sigmoid_close
        
        return np.clip(gamma, 0.0, gamma_max)
    
    def _gaussian(self, t_ms: float) -> float:
        """
        Profil gaussien : pic de plasticité centré sur t_peak.
        
        Formule: γ(t) = γ_max * exp(-(t-t_peak)²/(2σ²))
        
        Args:
            t_ms: Temps en millisecondes
            
        Returns:
            Valeur γ(t) pour fenêtre critique gaussienne
        """
        gamma_max = self.params['gamma_max']
        t_peak = self.params['t_peak_ms']
        sigma = self.params['sigma']
        
        # Gaussienne centrée sur t_peak
        gamma = gamma_max * np.exp(-((t_ms - t_peak) ** 2) / (2 * sigma ** 2))
        
        return np.clip(gamma, 0.0, gamma_max)
    
    def plot_profile(self, t_start_ms: float = 0, t_end_ms: float = 10000, 
                    n_points: int = 1000) -> tuple:
        """
        Génère les données pour tracer le profil γ(t).
        
        Args:
            t_start_ms: Temps de début
            t_end_ms: Temps de fin  
            n_points: Nombre de points
            
        Returns:
            (temps_array, gamma_array) pour plotting
        """
        times = np.linspace(t_start_ms, t_end_ms, n_points)
        gammas = np.array([self.value(t) for t in times])
        
        return times, gammas
    
    def get_description(self) -> str:
        """Retourne une description textuelle du profil."""
        if self.mode == 'double_sigmoid':
            return (f"Double-sigmoïde: ouverture à {self.params['t_open_ms']}ms, "
                   f"fermeture à {self.params['t_close_ms']}ms, "
                   f"γ_max={self.params['gamma_max']}")
        elif self.mode == 'gauss':
            return (f"Gaussien: pic à {self.params['t_peak_ms']}ms, "
                   f"σ={self.params['sigma']}ms, "
                   f"γ_max={self.params['gamma_max']}")
        else:
            return f"Mode {self.mode}"


def create_critical_window_scheduler(gamma_config: Dict[str, Any]) -> GammaScheduler:
    """
    Factory function pour créer un planificateur γ(t) depuis la config.
    
    Args:
        gamma_config: Configuration de la fenêtre critique
        
    Returns:
        Instance de GammaScheduler configurée
    """
    mode = gamma_config.get('mode', 'double_sigmoid')
    
    # Extraction des paramètres selon le mode
    if mode == 'double_sigmoid':
        params = {
            'gamma_max': gamma_config.get('gamma_max', 1.0),
            't_open_ms': gamma_config.get('t_open_ms', 1000),
            't_close_ms': gamma_config.get('t_close_ms', 5000),
            'slope_open': gamma_config.get('slope_open', 0.01),
            'slope_close': gamma_config.get('slope_close', 0.01)
        }
    elif mode == 'gauss':
        params = {
            'gamma_max': gamma_config.get('gamma_max', 1.0),
            't_peak_ms': gamma_config.get('t_peak_ms', 3000),
            'sigma': gamma_config.get('sigma', 1000)
        }
    else:
        raise ValueError(f"Mode γ(t) non supporté: {mode}")
    
    return GammaScheduler(mode, params) 