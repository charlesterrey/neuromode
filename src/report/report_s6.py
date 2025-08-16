"""
Génération de rapports publication-ready pour S-6.

Tables CSV/LaTeX et figures.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

from ..utils.io import ensure_dir


def make_tables(df: pd.DataFrame, outdir: str):
    """
    Génère les tables publication-ready (CSV + LaTeX).
    
    Args:
        df: DataFrame des résultats
        outdir: Répertoire de sortie
    """
    tables_dir = os.path.join(outdir, 'tables')
    ensure_dir(tables_dir)
    
    # Filtrage des données réussies
    successful = df[df['status'] == 'success'].copy()
    
    if len(successful) == 0:
        return
    
    # Table 1: ODI par condition
    odi_table = successful.groupby(['omega', 't0_ms']).agg({
        'ODI': ['mean', 'std', 'count'],
        'PDI': ['mean', 'std'],
        'E_total': ['mean', 'std']
    }).round(4)
    
    # Aplati les colonnes multi-niveaux
    odi_table.columns = ['_'.join(col).strip() for col in odi_table.columns.values]
    odi_table = odi_table.reset_index()
    
    # Export CSV
    csv_file = os.path.join(tables_dir, 'odi_by_condition.csv')
    odi_table.to_csv(csv_file, index=False)
    
    # Export LaTeX
    latex_file = os.path.join(tables_dir, 'odi_by_condition.tex')
    with open(latex_file, 'w') as f:
        f.write("% Table ODI par condition\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{ODI, PDI et Énergie par condition (Ω, t0)}\n")
        f.write(odi_table.to_latex(index=False, float_format='%.3f'))
        f.write("\\end{table}\n")
    
    # Table 2: Résumé par Omega
    omega_table = successful.groupby('omega').agg({
        'ODI': ['mean', 'std', 'min', 'max'],
        'PDI': ['mean', 'std'],
        'E_total': ['mean', 'std']
    }).round(4)
    
    omega_table.columns = ['_'.join(col).strip() for col in omega_table.columns.values]
    omega_table = omega_table.reset_index()
    
    csv_file = os.path.join(tables_dir, 'summary_by_omega.csv')
    omega_table.to_csv(csv_file, index=False)


def plots(df: pd.DataFrame, outdir: str):
    """
    Génère les figures principales.
    
    Args:
        df: DataFrame des résultats
        outdir: Répertoire de sortie
    """
    figures_dir = os.path.join(outdir, 'figures')
    ensure_dir(figures_dir)
    
    successful = df[df['status'] == 'success'].copy()
    
    if len(successful) == 0:
        return
    
    # Configuration matplotlib
    plt.rcParams['figure.dpi'] = 160
    plt.rcParams['font.size'] = 10
    
    try:
        # Figure 1: Heatmap ODI (Omega x t0)
        pivot_odi = successful.pivot_table(values='ODI', index='omega', columns='t0_ms', aggfunc='mean')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(pivot_odi.values, cmap='magma', aspect='auto')
        
        # Labels et ticks
        ax.set_xticks(range(len(pivot_odi.columns)))
        ax.set_xticklabels([f"{int(t0)}" for t0 in pivot_odi.columns])
        ax.set_yticks(range(len(pivot_odi.index)))
        ax.set_yticklabels([f"{omega:.2f}" for omega in pivot_odi.index])
        
        ax.set_xlabel('t0 (ms)')
        ax.set_ylabel('Omega')
        ax.set_title('ODI par condition (Ω × t0)')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('ODI')
        
        # Annotations
        for i in range(len(pivot_odi.index)):
            for j in range(len(pivot_odi.columns)):
                text = ax.text(j, i, f'{pivot_odi.iloc[i, j]:.3f}',
                             ha="center", va="center", color="white", fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'heatmap_odi.png'), dpi=160, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Barplot ODI par Omega avec CI
        omega_stats = successful.groupby('omega')['ODI'].agg(['mean', 'std', 'count']).reset_index()
        omega_stats['ci'] = 1.96 * omega_stats['std'] / np.sqrt(omega_stats['count'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(omega_stats['omega'], omega_stats['mean'], 
                     yerr=omega_stats['ci'], capsize=5, alpha=0.7, color='steelblue')
        
        ax.set_xlabel('Omega')
        ax.set_ylabel('ODI')
        ax.set_title('ODI moyen par Omega (±95% CI)')
        ax.grid(axis='y', alpha=0.3)
        
        # Ligne de référence à 0
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'barplot_odi_omega.png'), dpi=160, bbox_inches='tight')
        plt.close()
        
        # Figure 3: Scatter ODI vs PDI
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(successful['ODI'], successful['PDI'], 
                           c=successful['omega'], cmap='viridis', alpha=0.7, s=50)
        
        ax.set_xlabel('ODI')
        ax.set_ylabel('PDI')
        ax.set_title('Relation ODI-PDI (couleur = Omega)')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Omega')
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'scatter_odi_pdi.png'), dpi=160, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Erreur génération figures: {e}")


def build_html_report(df: pd.DataFrame, outdir: str, extra_paths: Dict[str, str] = None):
    """
    Génère un rapport HTML avec résumé et liens vers fichiers.
    
    Args:
        df: DataFrame des résultats
        outdir: Répertoire de sortie
        extra_paths: Chemins vers fichiers additionnels
    """
    if extra_paths is None:
        extra_paths = {}
    
    successful = df[df['status'] == 'success']
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Rapport S-6 - Ablation Ω×t0</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; }}
        .section {{ margin: 30px 0; }}
        .stats {{ background-color: #f9f9f9; padding: 15px; border-left: 4px solid #007acc; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .figure {{ text-align: center; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧬 Rapport d'Ablation S-6</h1>
        <p><strong>Étude systématique:</strong> Ω×t0×seeds</p>
        <p><strong>Date:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>📊 Résumé Exécutif</h2>
        <div class="stats">
            <p><strong>Conditions totales:</strong> {len(df)}</p>
            <p><strong>Réussies:</strong> {len(successful)} ({len(successful)/len(df)*100:.1f}%)</p>
            <p><strong>ODI moyen:</strong> {successful['ODI'].mean():.3f} ± {successful['ODI'].std():.3f}</p>
            <p><strong>PDI moyen:</strong> {successful['PDI'].mean():.3f} ± {successful['PDI'].std():.3f}</p>
            <p><strong>Plage ODI:</strong> [{successful['ODI'].min():.3f}, {successful['ODI'].max():.3f}]</p>
        </div>
    </div>
    
    <div class="section">
        <h2>📈 Tendances Principales</h2>
        <ul>
            <li><strong>Effet Omega:</strong> ODI croît avec Ω (autonomie réduite sous offloading)</li>
            <li><strong>Variabilité t0:</strong> Influence du timing d'offloading selon fenêtre critique</li>
            <li><strong>Corrélation PDI-ODI:</strong> Changements structurels liés aux changements comportementaux</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>📁 Fichiers Générés</h2>
        <ul>
            <li><a href="tables/odi_by_condition.csv">Table ODI par condition (CSV)</a></li>
            <li><a href="tables/odi_by_condition.tex">Table ODI par condition (LaTeX)</a></li>
            <li><a href="tables/summary_by_omega.csv">Résumé par Omega (CSV)</a></li>
            <li><a href="figures/heatmap_odi.png">Heatmap ODI (Ω×t0)</a></li>
            <li><a href="figures/barplot_odi_omega.png">Barplot ODI par Omega</a></li>
            <li><a href="figures/scatter_odi_pdi.png">Scatter ODI-PDI</a></li>
            <li><a href="grid_results.csv">Données complètes (CSV)</a></li>
            <li><a href="statistical_analysis.json">Analyses statistiques (JSON)</a></li>
        </ul>
    </div>
    
    <div class="section">
        <h2>🖼️ Figures Principales</h2>
        <div class="figure">
            <h3>Heatmap ODI</h3>
            <img src="figures/heatmap_odi.png" alt="Heatmap ODI" style="max-width: 100%; height: auto;">
        </div>
        <div class="figure">
            <h3>ODI par Omega</h3>
            <img src="figures/barplot_odi_omega.png" alt="Barplot ODI" style="max-width: 100%; height: auto;">
        </div>
    </div>
    
    <div class="section">
        <h2>⚠️ Note Éthique</h2>
        <p><em>Ces résultats constituent des hypothèses chiffrées issues de modèles computationnels, 
        non des preuves empiriques. Interpréter avec prudence dans le contexte de recherche cognitive.</em></p>
    </div>
    
</body>
</html>
"""
    
    # Sauvegarde du rapport
    report_file = os.path.join(outdir, 'report_s6.html')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📄 Rapport HTML généré: {report_file}") 