"""
Module de gestion énergétique pour contraintes métaboliques.

Implémente le suivi des coûts énergétiques (spikes, synapses, câblage)
et calcule la pression énergétique pour moduler l'élagage synaptique.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import os


class EnergyLedger:
    """
    Comptabilité énergétique pour contraintes métaboliques neurales.
    
    Suit les coûts de signalisation (spikes, événements synaptiques)
    et de maintenance (longueur de câblage) dans une fenêtre glissante.
    """
    
    def __init__(self, window_ms: float, c_spike: float, c_syn: float, 
                 c_len: float, budget_B: float):
        """
        Initialise le registre énergétique.
        
        Args:
            window_ms: Taille de fenêtre pour moyennage énergétique
            c_spike: Coût énergétique par spike
            c_syn: Coût énergétique par événement synaptique  
            c_len: Coût énergétique par unité de longueur de câblage
            budget_B: Budget énergétique total disponible
        """
        self.window_ms = window_ms
        self.c_spike = c_spike
        self.c_syn = c_syn
        self.c_len = c_len
        self.budget_B = budget_B
        
        # Historique pour calculs de fenêtre glissante
        self.history = []  # [(t_ms, E_spike, E_syn, E_len), ...]
        
    def update(self, t_ms: float, count_spikes: int, count_syn_events: int, 
              len_cost_sum: float) -> Tuple[float, float]:
        """
        Met à jour les coûts énergétiques et calcule la pression.
        
        Args:
            t_ms: Temps actuel en millisecondes
            count_spikes: Nombre de spikes dans l'intervalle
            count_syn_events: Nombre d'événements synaptiques 
            len_cost_sum: Somme des coûts de câblage actifs
            
        Returns:
            (E_win, P_E): Énergie de fenêtre et pression énergétique
        """
        # Calcul des coûts individuels
        E_spike = self.c_spike * count_spikes
        E_syn = self.c_syn * count_syn_events  
        E_len = self.c_len * len_cost_sum
        
        # Ajout à l'historique
        self.history.append((t_ms, E_spike, E_syn, E_len))
        
        # Élagage de l'historique (fenêtre glissante)
        cutoff_time = t_ms - self.window_ms
        self.history = [(t, es, esyn, el) for (t, es, esyn, el) in self.history 
                       if t >= cutoff_time]
        
        # Calcul de l'énergie totale dans la fenêtre
        if len(self.history) == 0:
            E_win = 0.0
        else:
            E_win = sum(es + esyn + el for (_, es, esyn, el) in self.history)
        
        # Calcul de la pression énergétique
        P_E = max(0.0, (E_win - self.budget_B) / self.budget_B) if self.budget_B > 0 else 0.0
        
        return E_win, P_E
    
    def get_current_breakdown(self) -> Dict[str, float]:
        """
        Retourne la décomposition énergétique actuelle.
        
        Returns:
            Dictionnaire avec breakdown des coûts
        """
        if len(self.history) == 0:
            return {
                'E_spike': 0.0,
                'E_syn': 0.0, 
                'E_len': 0.0,
                'E_total': 0.0,
                'budget': self.budget_B,
                'pressure': 0.0
            }
        
        E_spike_total = sum(es for (_, es, _, _) in self.history)
        E_syn_total = sum(esyn for (_, _, esyn, _) in self.history)
        E_len_total = sum(el for (_, _, _, el) in self.history)
        E_total = E_spike_total + E_syn_total + E_len_total
        
        pressure = max(0.0, (E_total - self.budget_B) / self.budget_B) if self.budget_B > 0 else 0.0
        
        return {
            'E_spike': E_spike_total,
            'E_syn': E_syn_total,
            'E_len': E_len_total, 
            'E_total': E_total,
            'budget': self.budget_B,
            'pressure': pressure
        }
    
    def export_history(self) -> List[Tuple[float, float, float, float, float, float]]:
        """
        Exporte l'historique pour sauvegarde.
        
        Returns:
            Liste de (t_ms, E_spike, E_syn, E_len, E_total, P_E)
        """
        exported = []
        
        for i, (t_ms, es, esyn, el) in enumerate(self.history):
            # Recalcul de la fenêtre à ce moment-là
            cutoff = t_ms - self.window_ms
            window_data = [(t, es_w, esyn_w, el_w) for (t, es_w, esyn_w, el_w) in self.history[:i+1] 
                          if t >= cutoff]
            
            if len(window_data) > 0:
                E_total = sum(es_w + esyn_w + el_w for (_, es_w, esyn_w, el_w) in window_data)
            else:
                E_total = 0.0
                
            P_E = max(0.0, (E_total - self.budget_B) / self.budget_B) if self.budget_B > 0 else 0.0
            
            exported.append((t_ms, es, esyn, el, E_total, P_E))
        
        return exported
    
    def save_to_csv(self, filepath: str) -> None:
        """
        Sauvegarde l'historique énergétique en CSV.
        
        Args:
            filepath: Chemin du fichier de sortie
        """
        import pandas as pd
        
        history_data = self.export_history()
        
        if len(history_data) == 0:
            # Fichier vide avec en-têtes
            df = pd.DataFrame(columns=['time_ms', 'E_spike', 'E_syn', 'E_len', 'E_total', 'P_E', 'budget'])
        else:
            data_rows = []
            for (t_ms, es, esyn, el, etot, pe) in history_data:
                data_rows.append({
                    'time_ms': t_ms,
                    'E_spike': es,
                    'E_syn': esyn,
                    'E_len': el,
                    'E_total': etot,
                    'P_E': pe,
                    'budget': self.budget_B
                })
            
            df = pd.DataFrame(data_rows)
        
        df.to_csv(filepath, index=False)
    
    def reset(self) -> None:
        """Remet à zéro l'historique énergétique."""
        self.history.clear()


def compute_synapse_activity_count(monitors: Dict[str, Any], 
                                 time_window_ms: float) -> int:
    """
    Compte les événements synaptiques E→E dans une fenêtre temporelle.
    
    Args:
        monitors: Dictionnaire des moniteurs Brian2
        time_window_ms: Fenêtre temporelle en ms
        
    Returns:
        Nombre d'événements synaptiques dans la fenêtre
    """
    try:
        # Accès aux spikes excitateurs
        spike_monitor = monitors.get('spikes_e')
        if spike_monitor is None or len(spike_monitor.t) == 0:
            return 0
        
        # Conversion en ms
        spike_times_ms = np.array(spike_monitor.t) * 1000  # Supposant que t est en secondes
        current_time_ms = spike_times_ms[-1] if len(spike_times_ms) > 0 else 0
        
        # Fenêtre récente
        cutoff_time = current_time_ms - time_window_ms
        recent_spikes = spike_times_ms[spike_times_ms >= cutoff_time]
        
        # Approximation : chaque spike excitateur déclenche ~densité*N_e événements synaptiques
        syn_ee = monitors.get('syn_ee')
        if syn_ee is not None:
            density = np.mean(syn_ee.alive) if hasattr(syn_ee, 'alive') else 0.1
            N_e = len(monitors.get('neurons_e', []))
            events_per_spike = density * N_e
            
            return int(len(recent_spikes) * events_per_spike)
        else:
            return len(recent_spikes)  # Fallback
            
    except Exception:
        return 0  # Gestion d'erreur gracieuse


def compute_total_wiring_cost(monitors: Dict[str, Any]) -> float:
    """
    Calcule le coût total de câblage des connexions actives.
    
    Args:
        monitors: Dictionnaire des moniteurs Brian2
        
    Returns:
        Somme des coûts de câblage pour connexions actives
    """
    try:
        syn_ee = monitors.get('syn_ee')
        if syn_ee is None:
            return 0.0
        
        if hasattr(syn_ee, 'alive') and hasattr(syn_ee, 'len_cost'):
            alive_mask = np.array(syn_ee.alive) > 0.5
            len_costs = np.array(syn_ee.len_cost)
            
            return float(np.sum(len_costs[alive_mask]))
        else:
            return 0.0
            
    except Exception:
        return 0.0  # Gestion d'erreur gracieuse


def create_energy_tracker(energy_config: Dict[str, Any]) -> EnergyLedger:
    """
    Factory function pour créer un tracker énergétique depuis la config.
    
    Args:
        energy_config: Configuration énergétique
        
    Returns:
        Instance d'EnergyLedger configurée
    """
    return EnergyLedger(
        window_ms=energy_config.get('window_ms', 200),
        c_spike=energy_config.get('c_spike', 1.0),
        c_syn=energy_config.get('c_syn', 0.1),
        c_len=energy_config.get('c_len', 0.01),
        budget_B=energy_config.get('budget_B', 5000.0)
    ) 