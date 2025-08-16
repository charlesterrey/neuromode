"""
Module de planification de l'offloading et contrôle LC-NE.

Implémente la variable d'offloading Ω(t) qui réduit l'effort endogène
et module la neuromodulation LC-NE g_NE(t) pour diminuer la plasticité.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import json


class OffloadingSchedule:
    """
    Planificateur de l'offloading mémoire Ω(t).
    
    Contrôle la transition entre effort endogène maximal (Ω=0) 
    et délestage vers structures externes (Ω→1), modélisé par
    une fonction sigmoïde centrée sur t0.
    """
    
    def __init__(self, omega: float, t0_ms: float, sigma_ms: float = 200):
        """
        Initialise la planification d'offloading.
        
        Args:
            omega: Niveau maximal d'offloading [0,1]
            t0_ms: Temps de début du processus d'offloading
            sigma_ms: Largeur de la transition sigmoïde
        """
        self.omega = np.clip(omega, 0.0, 1.0)
        self.t0_ms = t0_ms
        self.sigma_ms = sigma_ms
        
        # Pente de la sigmoïde (plus sigma petit = transition plus brutale)
        self.slope = 1.0 / sigma_ms if sigma_ms > 0 else 10.0
    
    def effort(self, t_ms: float) -> float:
        """
        Calcule l'effort endogène à l'instant t.
        
        Formule: effort(t) = 1 - Ω * σ((t-t0)/σ)
        
        Args:
            t_ms: Temps en millisecondes
            
        Returns:
            Niveau d'effort dans [1-Ω, 1] (1=max effort, 1-Ω=min effort)
        """
        if t_ms < self.t0_ms:
            return 1.0  # Effort maximal avant t0
        
        # Sigmoïde d'offloading après t0
        sigmoid_val = 1.0 / (1.0 + np.exp(-self.slope * (t_ms - self.t0_ms)))
        effort_level = 1.0 - self.omega * sigmoid_val
        
        return np.clip(effort_level, 1.0 - self.omega, 1.0)
    
    def get_curve(self, t_start_ms: float = 0, t_end_ms: float = 10000, 
                  n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Génère la courbe effort(t) pour plotting.
        
        Returns:
            (temps_array, effort_array)
        """
        times = np.linspace(t_start_ms, t_end_ms, n_points)
        efforts = np.array([self.effort(t) for t in times])
        
        return times, efforts
    
    def get_description(self) -> str:
        """Description textuelle de la planification."""
        return (f"Offloading: Ω={self.omega:.2f}, t0={self.t0_ms:.0f}ms, "
               f"σ={self.sigma_ms:.0f}ms")


class NEController:
    """
    Contrôleur LC-NE (Locus Coeruleus - Noradrénaline).
    
    Module la neuromodulation en fonction de l'effort et génère
    des pulses phasiques aléatoires sur un niveau tonique.
    """
    
    def __init__(self, g0: float, tonic: float, phasic_rate_hz: float, 
                 phasic_amp: float, seed: int = 42):
        """
        Initialise le contrôleur LC-NE.
        
        Args:
            g0: Gain baseline maximum
            tonic: Niveau tonique minimal [0,1]
            phasic_rate_hz: Fréquence des pulses phasiques
            phasic_amp: Amplitude des pulses phasiques
            seed: Graine pour génération aléatoire
        """
        self.g0 = g0
        self.tonic = np.clip(tonic, 0.0, 1.0)
        self.phasic_rate_hz = phasic_rate_hz
        self.phasic_amp = phasic_amp
        
        # Générateur RNG pour pulses phasiques
        self.rng = np.random.default_rng(seed)
        
        # Historique des pulses phasiques (temps_ms, amplitude)
        self.phasic_events = []
    
    def _generate_phasic_pulse(self, t_ms: float, dt_ms: float) -> float:
        """
        Génère un pulse phasique aléatoire selon un processus de Poisson.
        
        Args:
            t_ms: Temps actuel
            dt_ms: Intervalle temporel
            
        Returns:
            Amplitude du pulse (0 si pas de pulse)
        """
        # Probabilité de pulse dans l'intervalle dt
        prob_pulse = self.phasic_rate_hz * (dt_ms / 1000.0)
        
        if self.rng.random() < prob_pulse:
            # Génération d'un pulse avec amplitude variable
            amp = self.phasic_amp * (0.5 + 0.5 * self.rng.random())
            self.phasic_events.append((t_ms, amp))
            return amp
        else:
            return 0.0
    
    def value(self, t_ms: float, effort_level: float, dt_ms: float = 50) -> float:
        """
        Calcule g_NE(t) en fonction de l'effort et des pulses phasiques.
        
        Formule: g_NE(t) = g0 * effort * (tonic + phasic_pulse(t))
        
        Args:
            t_ms: Temps actuel
            effort_level: Niveau d'effort [0,1] depuis OffloadingSchedule
            dt_ms: Intervalle pour génération pulses
            
        Returns:
            Gain neuromodulateur g_NE(t)
        """
        # Pulse phasique aléatoire
        phasic_component = self._generate_phasic_pulse(t_ms, dt_ms)
        
        # Calcul du gain total
        ne_level = self.tonic + phasic_component
        g_ne = self.g0 * effort_level * ne_level
        
        return np.clip(g_ne, 0.0, self.g0 * 2.0)  # Bornage sécuritaire
    
    def get_phasic_history(self) -> List[Tuple[float, float]]:
        """Retourne l'historique des pulses phasiques."""
        return self.phasic_events.copy()
    
    def reset_history(self) -> None:
        """Remet à zéro l'historique des pulses."""
        self.phasic_events.clear()
    
    def get_description(self) -> str:
        """Description textuelle du contrôleur."""
        return (f"LC-NE: g0={self.g0:.2f}, tonique={self.tonic:.2f}, "
               f"phasique={self.phasic_rate_hz:.1f}Hz @ {self.phasic_amp:.2f}")


def create_offloading_scheduler(offloading_config: Dict[str, Any]) -> OffloadingSchedule:
    """
    Factory function pour créer un planificateur d'offloading depuis la config.
    
    Args:
        offloading_config: Configuration d'offloading
        
    Returns:
        Instance d'OffloadingSchedule configurée
    """
    return OffloadingSchedule(
        omega=offloading_config.get('omega', 0.5),
        t0_ms=offloading_config.get('t0_ms', 4000),
        sigma_ms=offloading_config.get('sigma_ms', 300)
    )


def create_ne_controller(lc_ne_config: Dict[str, Any], seed: int = 42) -> NEController:
    """
    Factory function pour créer un contrôleur LC-NE depuis la config.
    
    Args:
        lc_ne_config: Configuration LC-NE
        seed: Graine pour reproductibilité
        
    Returns:
        Instance de NEController configurée
    """
    return NEController(
        g0=lc_ne_config.get('g0', 1.0),
        tonic=lc_ne_config.get('tonic', 0.2),
        phasic_rate_hz=lc_ne_config.get('phasic_rate_hz', 3.0),
        phasic_amp=lc_ne_config.get('phasic_amp', 0.8),
        seed=seed
    )


def compute_odi(recall_noassist: float, recall_assist: float) -> float:
    """
    Calcule l'Offloading Dependency Index (ODI).
    
    ODI = recall_noassist - recall_assist
    
    Un ODI positif indique une dépendance à l'aide externe
    (performance dégradée sans assistance).
    
    Args:
        recall_noassist: Score de rappel sans assistance
        recall_assist: Score de rappel avec assistance
        
    Returns:
        Index ODI (plus élevé = plus de dépendance)
    """
    return recall_noassist - recall_assist


def export_offloading_curves(offloading_schedule: OffloadingSchedule,
                           ne_controller: NEController,
                           duration_ms: float,
                           dt_ms: float = 50,
                           filepath: str = "offloading_curves.json") -> Dict[str, Any]:
    """
    Exporte les courbes d'effort et g_NE pour analyse.
    
    Args:
        offloading_schedule: Planificateur d'offloading
        ne_controller: Contrôleur LC-NE
        duration_ms: Durée de simulation
        dt_ms: Pas temporel
        filepath: Chemin de sortie JSON
        
    Returns:
        Dictionnaire avec les courbes
    """
    times = np.arange(0, duration_ms, dt_ms)
    efforts = []
    g_nes = []
    
    for t in times:
        effort = offloading_schedule.effort(t)
        g_ne = ne_controller.value(t, effort, dt_ms)
        
        efforts.append(effort)
        g_nes.append(g_ne)
    
    curves_data = {
        'times_ms': times.tolist(),
        'effort_levels': efforts,
        'gne_values': g_nes,
        'phasic_events': ne_controller.get_phasic_history(),
        'metadata': {
            'offloading': offloading_schedule.get_description(),
            'lc_ne': ne_controller.get_description(),
            'duration_ms': duration_ms,
            'dt_ms': dt_ms
        }
    }
    
    # Sauvegarde JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(curves_data, f, indent=2, ensure_ascii=False)
    
    return curves_data 