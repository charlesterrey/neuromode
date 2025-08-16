#!/usr/bin/env python3
"""
Script d'analyse interactive des résultats neuro_offload_model.

Usage:
    python analyze_results.py [chemin_vers_resultats]
    
Exemples:
    python analyze_results.py outputs/test_s2/
    python analyze_results.py outputs/test_s3/
    python analyze_results.py test1/s6_ablation/
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional

# Import des modules d'analyse du modèle
sys.path.append('src')
from src.metrics.behavior import compute_ODI, bootstrap_ci, summarize_odi_by_condition
from src.metrics.energy import energy_aggregates, summarize_energy_by_condition
from src.metrics.structure import structural_summary, degree_stats
from src.stats.stats import anova_two_way, perm_test_diff, corr_spearman
from src.utils.io import log_info, log_error


class ResultsAnalyzer:
    """Analyseur de résultats pour neuro_offload_model."""
    
    def __init__(self, results_dir: str):
        """
        Initialise l'analyseur.
        
        Args:
            results_dir: Répertoire contenant les résultats
        """
        self.results_dir = Path(results_dir)
        self.model_type = self._detect_model_type()
        self.data = {}
        
        log_info(f"Analyse du répertoire: {self.results_dir}")
        log_info(f"Type de modèle détecté: {self.model_type}")
    
    def _detect_model_type(self) -> str:
        """Détecte le type de modèle basé sur les fichiers présents."""
        files = list(self.results_dir.glob("*"))
        
        if any("grid_results" in f.name for f in files):
            return "S6_ablation"
        elif any("energy.csv" in f.name for f in files):
            return "S3_energy" if any("gamma.csv" in f.name for f in files) else "S4_offloading"
        elif any("structural_events.csv" in f.name for f in files):
            return "S2_structural"
        elif any("spikes_e.csv" in f.name for f in files):
            return "S1_basic"
        else:
            return "unknown"
    
    def load_data(self):
        """Charge les données selon le type de modèle."""
        try:
            if self.model_type == "S6_ablation":
                self._load_s6_data()
            elif self.model_type in ["S3_energy", "S4_offloading"]:
                self._load_energy_data()
            elif self.model_type == "S2_structural":
                self._load_structural_data()
            elif self.model_type == "S1_basic":
                self._load_basic_data()
            else:
                log_error(f"Type de modèle non reconnu: {self.model_type}")
                
        except Exception as e:
            log_error(f"Erreur lors du chargement: {e}")
    
    def _load_s6_data(self):
        """Charge les données S6 (études d'ablation)."""
        csv_file = self.results_dir / "grid_results.csv"
        if csv_file.exists():
            self.data['grid'] = pd.read_csv(csv_file)
            log_info(f"Données S6 chargées: {len(self.data['grid'])} conditions")
    
    def _load_energy_data(self):
        """Charge les données énergétiques (S3/S4)."""
        energy_file = self.results_dir / "energy.csv"
        if energy_file.exists():
            self.data['energy'] = pd.read_csv(energy_file)
            
        # ODI pour S4
        odi_file = self.results_dir / "odi.json"
        if odi_file.exists():
            with open(odi_file, 'r') as f:
                self.data['odi'] = json.load(f)
        
        # Gamma pour S3
        gamma_file = self.results_dir / "gamma.csv"
        if gamma_file.exists():
            self.data['gamma'] = pd.read_csv(gamma_file)
    
    def _load_structural_data(self):
        """Charge les données structurelles (S2)."""
        # Événements structurels
        events_file = self.results_dir / "structural_events.csv"
        if events_file.exists():
            self.data['events'] = pd.read_csv(events_file)
        
        # Évolution de densité
        density_file = self.results_dir / "density_evolution.csv"
        if density_file.exists():
            self.data['density'] = pd.read_csv(density_file)
        
        # Matrices d'adjacence
        adj_files = list(self.results_dir.glob("adjacency_t*.npy"))
        if adj_files:
            self.data['adjacency'] = {}
            for f in adj_files:
                time_point = f.name.replace('adjacency_t', '').replace('.npy', '')
                self.data['adjacency'][time_point] = np.load(f)
    
    def _load_basic_data(self):
        """Charge les données de base (S1)."""
        # Spikes
        spikes_e_file = self.results_dir / "spikes_e.csv"
        if spikes_e_file.exists():
            self.data['spikes_e'] = pd.read_csv(spikes_e_file)
        
        # Taux de décharge
        rates_e_file = self.results_dir / "rates_e.csv"
        if rates_e_file.exists():
            self.data['rates_e'] = pd.read_csv(rates_e_file)
    
    def analyze(self):
        """Lance l'analyse complète selon le type de modèle."""
        print(f"\n🔬 ANALYSE {self.model_type.upper()}")
        print("=" * 50)
        
        if self.model_type == "S6_ablation":
            self._analyze_s6()
        elif self.model_type in ["S3_energy", "S4_offloading"]:
            self._analyze_energy()
        elif self.model_type == "S2_structural":
            self._analyze_structural()
        elif self.model_type == "S1_basic":
            self._analyze_basic()
    
    def _analyze_s6(self):
        """Analyse des études d'ablation S6."""
        if 'grid' not in self.data:
            return
        
        df = self.data['grid']
        successful = df[df['status'] == 'success']
        
        print(f"📊 Conditions réussies: {len(successful)}/{len(df)}")
        
        if len(successful) == 0:
            return
        
        # Statistiques ODI
        print(f"\n📈 STATISTIQUES ODI:")
        print(f"   Moyenne: {successful['ODI'].mean():.3f}")
        print(f"   Écart-type: {successful['ODI'].std():.3f}")
        print(f"   Min/Max: {successful['ODI'].min():.3f} / {successful['ODI'].max():.3f}")
        
        # Analyse par Omega
        print(f"\n🔧 ANALYSE PAR OMEGA:")
        for omega in sorted(successful['omega'].unique()):
            subset = successful[successful['omega'] == omega]
            print(f"   Ω={omega}: ODI={subset['ODI'].mean():.3f}±{subset['ODI'].std():.3f} (n={len(subset)})")
        
        # Tests statistiques
        if len(successful['omega'].unique()) > 1:
            try:
                from scipy.stats import f_oneway
                groups = [group['ODI'].values for _, group in successful.groupby('omega')]
                f_stat, p_val = f_oneway(*groups)
                print(f"\n📊 ANOVA Omega: F={f_stat:.2f}, p={p_val:.4f}")
                print(f"   {'Significatif' if p_val < 0.05 else 'Non significatif'} (α=0.05)")
            except:
                pass
    
    def _analyze_energy(self):
        """Analyse énergétique (S3/S4)."""
        if 'energy' in self.data:
            energy_df = self.data['energy']
            print(f"📊 Données énergétiques: {len(energy_df)} points temporels")
            
            if 'E_win' in energy_df.columns:
                print(f"   Énergie totale: {energy_df['E_win'].sum():.0f}")
                print(f"   Énergie moyenne/fenêtre: {energy_df['E_win'].mean():.1f}±{energy_df['E_win'].std():.1f}")
                print(f"   Pic énergétique: {energy_df['E_win'].max():.1f}")
        
        if 'odi' in self.data:
            odi_data = self.data['odi']
            print(f"\n🧠 OFFLOADING (ODI):")
            print(f"   Rappel sans aide: {odi_data.get('recall_noassist', 'N/A')}")
            print(f"   Rappel avec aide: {odi_data.get('recall_assist', 'N/A')}")
            print(f"   ODI: {odi_data.get('odi', 'N/A')}")
            print(f"   Interprétation: {odi_data.get('interpretation', 'N/A')}")
        
        if 'gamma' in self.data:
            gamma_df = self.data['gamma']
            print(f"\n⚡ MODULATION GAMMA:")
            # Détecter la colonne gamma (peut être 'gamma' ou autre nom)
            gamma_col = None
            for col in gamma_df.columns:
                if 'gamma' in col.lower():
                    gamma_col = col
                    break
            
            if gamma_col:
                print(f"   Valeurs: {gamma_df[gamma_col].min():.2f} - {gamma_df[gamma_col].max():.2f}")
                print(f"   Moyenne: {gamma_df[gamma_col].mean():.3f}±{gamma_df[gamma_col].std():.3f}")
            else:
                print(f"   Colonnes disponibles: {list(gamma_df.columns)}")
    
    def _analyze_structural(self):
        """Analyse structurelle (S2)."""
        if 'events' in self.data:
            events_df = self.data['events']
            print(f"📊 Événements structurels: {len(events_df)} événements")
            
            if 'event_type' in events_df.columns:
                type_counts = events_df['event_type'].value_counts()
                for event_type, count in type_counts.items():
                    print(f"   {event_type}: {count}")
        
        if 'density' in self.data:
            density_df = self.data['density']
            print(f"\n🌐 ÉVOLUTION DENSITÉ:")
            print(f"   Points temporels: {len(density_df)}")
            if 'density' in density_df.columns:
                print(f"   Densité initiale: {density_df['density'].iloc[0]:.4f}")
                print(f"   Densité finale: {density_df['density'].iloc[-1]:.4f}")
                print(f"   Changement: {(density_df['density'].iloc[-1] - density_df['density'].iloc[0]):.4f}")
        
        if 'adjacency' in self.data:
            print(f"\n🔗 MATRICES D'ADJACENCE:")
            print(f"   Points temporels: {len(self.data['adjacency'])}")
            
            # Analyse du premier et dernier point
            time_points = sorted(self.data['adjacency'].keys(), key=int)
            if len(time_points) >= 2:
                A_start = self.data['adjacency'][time_points[0]]
                A_end = self.data['adjacency'][time_points[-1]]
                
                print(f"   t={time_points[0]}: {A_start.sum():.0f} connexions")
                print(f"   t={time_points[-1]}: {A_end.sum():.0f} connexions")
    
    def _analyze_basic(self):
        """Analyse de base (S1)."""
        if 'spikes_e' in self.data:
            spikes_df = self.data['spikes_e']
            print(f"📊 Spikes excitateurs: {len(spikes_df)} événements")
            
            if 'time_ms' in spikes_df.columns:
                duration = spikes_df['time_ms'].max() - spikes_df['time_ms'].min()
                rate = len(spikes_df) / (duration / 1000)  # Hz
                print(f"   Durée: {duration:.0f} ms")
                print(f"   Taux global: {rate:.1f} Hz")
        
        if 'rates_e' in self.data:
            rates_df = self.data['rates_e']
            print(f"\n📈 TAUX DE DÉCHARGE:")
            print(f"   Points temporels: {len(rates_df)}")
            if len(rates_df.columns) > 1:  # Colonnes de neurones
                neuron_cols = [col for col in rates_df.columns if col != 'time_ms']
                rates_matrix = rates_df[neuron_cols].values
                print(f"   Neurones: {len(neuron_cols)}")
                print(f"   Taux moyen: {rates_matrix.mean():.2f}±{rates_matrix.std():.2f} Hz")
    
    def plot_summary(self, save_dir: Optional[str] = None):
        """Génère des graphiques de résumé."""
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True)
        else:
            save_path = self.results_dir / "analysis_plots"
            save_path.mkdir(exist_ok=True)
        
        plt.style.use('default')
        
        if self.model_type == "S6_ablation" and 'grid' in self.data:
            self._plot_s6_summary(save_path)
        elif self.model_type in ["S3_energy", "S4_offloading"] and 'energy' in self.data:
            self._plot_energy_summary(save_path)
        elif self.model_type == "S2_structural" and 'density' in self.data:
            self._plot_structural_summary(save_path)
        elif self.model_type == "S1_basic" and 'rates_e' in self.data:
            self._plot_basic_summary(save_path)
        
        print(f"📊 Graphiques sauvegardés dans: {save_path}")
    
    def _plot_s6_summary(self, save_path: Path):
        """Graphiques de résumé S6."""
        df = self.data['grid']
        successful = df[df['status'] == 'success']
        
        if len(successful) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Boxplot ODI par Omega
        if len(successful['omega'].unique()) > 1:
            successful.boxplot(column='ODI', by='omega', ax=axes[0, 0])
            axes[0, 0].set_title('ODI par Omega')
            axes[0, 0].set_xlabel('Omega')
            axes[0, 0].set_ylabel('ODI')
        
        # 2. Scatter ODI vs PDI
        if 'PDI' in successful.columns:
            axes[0, 1].scatter(successful['ODI'], successful['PDI'], alpha=0.6)
            axes[0, 1].set_xlabel('ODI')
            axes[0, 1].set_ylabel('PDI')
            axes[0, 1].set_title('ODI vs PDI')
        
        # 3. Distribution ODI
        axes[1, 0].hist(successful['ODI'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('ODI')
        axes[1, 0].set_ylabel('Fréquence')
        axes[1, 0].set_title('Distribution ODI')
        
        # 4. Énergie vs ODI
        if 'E_total' in successful.columns:
            axes[1, 1].scatter(successful['E_total'], successful['ODI'], alpha=0.6)
            axes[1, 1].set_xlabel('Énergie totale')
            axes[1, 1].set_ylabel('ODI')
            axes[1, 1].set_title('Énergie vs ODI')
        
        plt.tight_layout()
        plt.savefig(save_path / 'summary_s6.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_energy_summary(self, save_path: Path):
        """Graphiques de résumé énergétique."""
        energy_df = self.data['energy']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Détecter les colonnes disponibles
        time_col = None
        energy_col = None
        pressure_col = None
        
        for col in energy_df.columns:
            if 't' in col.lower() or 'time' in col.lower():
                time_col = col
            elif 'E_win' in col or 'energy' in col.lower():
                energy_col = col
            elif 'P_E' in col or 'pressure' in col.lower():
                pressure_col = col
        
        # Si pas de colonne temps, créer un index
        if time_col is None:
            time_data = range(len(energy_df))
            time_label = 'Points temporels'
        else:
            time_data = energy_df[time_col]
            time_label = 'Temps (ms)'
        
        # 1. Évolution temporelle énergie
        if energy_col:
            axes[0, 0].plot(time_data, energy_df[energy_col])
            axes[0, 0].set_xlabel(time_label)
            axes[0, 0].set_ylabel('Énergie fenêtre')
            axes[0, 0].set_title('Évolution énergétique')
        else:
            axes[0, 0].text(0.5, 0.5, 'Données énergie non trouvées', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Évolution énergétique')
        
        # 2. Distribution énergie
        if energy_col:
            axes[0, 1].hist(energy_df[energy_col], bins=30, alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Énergie fenêtre')
            axes[0, 1].set_ylabel('Fréquence')
            axes[0, 1].set_title('Distribution énergétique')
        else:
            axes[0, 1].text(0.5, 0.5, 'Données énergie non trouvées', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Distribution énergétique')
        
        # 3. Pression énergétique
        if pressure_col:
            axes[1, 0].plot(time_data, energy_df[pressure_col], color='red')
            axes[1, 0].set_xlabel(time_label)
            axes[1, 0].set_ylabel('Pression P_E')
            axes[1, 0].set_title('Pression énergétique')
        else:
            axes[1, 0].text(0.5, 0.5, f'Colonnes: {list(energy_df.columns)}', ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=8)
            axes[1, 0].set_title('Pression énergétique')
        
        # 4. Gamma (si disponible)
        if 'gamma' in self.data:
            gamma_df = self.data['gamma']
            # Utiliser les premières colonnes numériques
            if len(gamma_df.columns) > 0:
                first_col = gamma_df.columns[0]
                if len(gamma_df.columns) > 1:
                    second_col = gamma_df.columns[1]
                    axes[1, 1].plot(gamma_df[first_col], gamma_df[second_col], color='purple')
                    axes[1, 1].set_xlabel(f'Colonne {first_col}')
                    axes[1, 1].set_ylabel(f'Colonne {second_col}')
                else:
                    axes[1, 1].plot(gamma_df[first_col], color='purple')
                    axes[1, 1].set_ylabel(f'Colonne {first_col}')
                axes[1, 1].set_title('Modulation Gamma')
        else:
            axes[1, 1].text(0.5, 0.5, 'Pas de données gamma', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Modulation Gamma')
        
        plt.tight_layout()
        plt.savefig(save_path / 'summary_energy.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_structural_summary(self, save_path: Path):
        """Graphiques de résumé structurel."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. Évolution densité
        if 'density' in self.data:
            density_df = self.data['density']
            if 't_ms' in density_df.columns:
                axes[0, 0].plot(density_df['t_ms'], density_df['density'])
                axes[0, 0].set_xlabel('Temps (ms)')
                axes[0, 0].set_ylabel('Densité')
                axes[0, 0].set_title('Évolution de la densité')
        
        # 2. Événements structurels
        if 'events' in self.data:
            events_df = self.data['events']
            if 'event_type' in events_df.columns:
                type_counts = events_df['event_type'].value_counts()
                axes[0, 1].bar(range(len(type_counts)), type_counts.values)
                axes[0, 1].set_xticks(range(len(type_counts)))
                axes[0, 1].set_xticklabels(type_counts.index, rotation=45)
                axes[0, 1].set_ylabel('Nombre d\'événements')
                axes[0, 1].set_title('Types d\'événements')
        
        plt.tight_layout()
        plt.savefig(save_path / 'summary_structural.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_basic_summary(self, save_path: Path):
        """Graphiques de résumé basique."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Évolution des taux de décharge
        if 'rates_e' in self.data:
            rates_df = self.data['rates_e']
            if 'time_ms' in rates_df.columns:
                neuron_cols = [col for col in rates_df.columns if col != 'time_ms']
                if neuron_cols:
                    # Moyenner sur quelques neurones pour visualisation
                    sample_neurons = neuron_cols[:min(10, len(neuron_cols))]
                    for neuron in sample_neurons:
                        axes[0, 0].plot(rates_df['time_ms'], rates_df[neuron], alpha=0.7)
                    axes[0, 0].set_xlabel('Temps (ms)')
                    axes[0, 0].set_ylabel('Taux de décharge (Hz)')
                    axes[0, 0].set_title('Évolution des taux (échantillon)')
        
        plt.tight_layout()
        plt.savefig(save_path / 'summary_basic.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Fonction principale."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyse des résultats neuro_offload_model")
    parser.add_argument('results_dir', nargs='?', default='outputs/test_s2', 
                       help='Répertoire des résultats à analyser')
    parser.add_argument('--plot', action='store_true', help='Générer les graphiques')
    parser.add_argument('--save-plots', type=str, help='Répertoire pour sauvegarder les graphiques')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"❌ Répertoire non trouvé: {args.results_dir}")
        print("\n💡 Répertoires disponibles:")
        for d in ['outputs', 'test1']:
            if os.path.exists(d):
                subdirs = [f for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))]
                for subdir in subdirs[:5]:  # Limiter l'affichage
                    print(f"   {d}/{subdir}/")
        return 1
    
    # Analyse
    analyzer = ResultsAnalyzer(args.results_dir)
    analyzer.load_data()
    analyzer.analyze()
    
    # Graphiques
    if args.plot:
        analyzer.plot_summary(args.save_plots)
    
    return 0


if __name__ == '__main__':
    sys.exit(main()) 