# 🔬 Guide d'Analyse des Résultats - neuro_offload_model

## Vue d'ensemble

Ce guide vous explique comment analyser les résultats de votre modèle de déchargement cognitif neuronal. Le modèle génère différents types de données selon le niveau de simulation (S1-S6).

## 🗂️ Structure des Résultats

### Répertoires de sortie

```
outputs/
├── test_s2/          # Plasticité structurelle
├── test_s3/          # Fenêtres critiques + énergie
├── run_YYYYMMDD_*    # Simulations horodatées
└── ...

test1/
├── s2_results/       # Résultats S2 (plasticité)
├── s4_results/       # Résultats S4 (offloading)
├── s6_ablation/      # Études d'ablation
└── ...
```

### Types de fichiers générés

| Type | Description | Modèles |
|------|-------------|---------|
| `spikes_e.csv` | Trains de spikes excitateurs | S1, S2, S3, S4 |
| `rates_e.csv` | Taux de décharge dans le temps | S1, S2, S3, S4 |
| `structural_events.csv` | Événements de croissance/élagage | S2, S3, S4 |
| `adjacency_t*.npy` | Matrices de connectivité | S2, S3, S4 |
| `energy.csv` | Consommation énergétique | S3, S4 |
| `odi.json` | Indices de déchargement (ODI) | S4 |
| `grid_results.csv` | Résultats d'ablation | S6 |
| `figures/*.png` | Visualisations automatiques | Tous |

## 🛠️ Utilisation du Script d'Analyse

### Installation des dépendances

```bash
# Dans le répertoire du projet
pip install pandas matplotlib seaborn scipy
```

### Usage de base

```bash
# Analyse simple
python3 analyze_results.py outputs/test_s2/

# Analyse avec graphiques
python3 analyze_results.py outputs/test_s3/ --plot

# Sauvegarder les graphiques ailleurs
python3 analyze_results.py test1/s6_ablation/ --plot --save-plots ~/Desktop/plots/
```

## 📊 Types d'Analyses par Modèle

### S1 : Modèle de Base (LIF + STDP)

**Données disponibles :**
- Activité neuronale (`spikes_e.csv`, `rates_e.csv`)
- Évolution des poids synaptiques

**Analyses générées :**
- Taux de décharge global
- Évolution temporelle de l'activité
- Statistiques de base

**Exemple d'interprétation :**
```
📊 Spikes excitateurs: 45,832 événements
   Durée: 5000 ms
   Taux global: 9.2 Hz

📈 TAUX DE DÉCHARGE:
   Points temporels: 500
   Neurones: 400
   Taux moyen: 8.45±3.21 Hz
```

### S2 : Plasticité Structurelle

**Données disponibles :**
- Événements structurels (`structural_events.csv`)
- Évolution de la densité (`density_evolution.csv`)
- Matrices d'adjacence temporelles (`adjacency_t*.npy`)

**Analyses générées :**
- Comptage des événements GROW/PRUNE
- Évolution de la densité de connectivité
- Comparaison structurelle temporelle

**Exemple d'interprétation :**
```
📊 Événements structurels: 27 événements
   GROW: 16 (nouvelles connexions)
   PRUNE: 11 (élagage)

🌐 ÉVOLUTION DENSITÉ:
   Densité initiale: 0.0312
   Densité finale: 0.0000
   Changement: -0.0312 (élagage net)
```

### S3 : Fenêtres Critiques + Énergie

**Données disponibles :**
- Consommation énergétique (`energy.csv`)
- Modulation gamma (`gamma.csv`)
- Données structurelles (hérité de S2)

**Analyses générées :**
- Profil énergétique temporel
- Statistiques de consommation
- Analyse des pics énergétiques

**Exemple d'interprétation :**
```
📊 Données énergétiques: 200 points temporels
   Énergie totale: 7,148,185
   Énergie moyenne/fenêtre: 35,740.9±67,626.7
   Pic énergétique: 216,180.9

⚡ Interprétation: Consommation variable avec des pics
   indiquant des périodes d'activité intense
```

### S4 : Déchargement Cognitif

**Données disponibles :**
- Indices ODI (`odi.json`)
- Données énergétiques (hérité de S3)
- Métriques comportementales

**Analyses générées :**
- Performance avec/sans assistance
- Calcul de l'ODI (Offloading Dependency Index)
- Corrélations énergie-comportement

**Exemple d'interprétation :**
```
🧠 OFFLOADING (ODI):
   Rappel sans aide: 0.75
   Rappel avec aide: 0.85
   ODI: 0.10
   Interprétation: Dépendance modérée à l'aide externe

🔍 Signification ODI:
   - ODI > 0 : Dépendance à l'assistance
   - ODI ≈ 0 : Autonomie préservée
   - ODI < 0 : Performance dégradée avec aide
```

### S6 : Études d'Ablation

**Données disponibles :**
- Grille de résultats (`grid_results.csv`)
- Analyses statistiques
- Rapports HTML

**Analyses générées :**
- Comparaisons entre conditions
- Tests statistiques (ANOVA, permutations)
- Effets principaux et interactions

**Exemple d'interprétation :**
```
📊 Conditions réussies: 24/30

📈 STATISTIQUES ODI:
   Moyenne: 0.068±0.110
   Range: -0.010 à 0.146

🔧 ANALYSE PAR OMEGA:
   Ω=0.0: ODI=-0.010 (contrôle)
   Ω=0.5: ODI=0.146 (modulation)
   Ω=1.0: ODI=0.089 (saturation)

📊 ANOVA: F=12.4, p=0.002 (significatif)
```

## 📈 Graphiques Générés

### Types de visualisations

1. **S2 (Structurel) :**
   - Évolution de la densité de connectivité
   - Distribution des événements structurels

2. **S3/S4 (Énergie) :**
   - Série temporelle énergétique
   - Distribution des consommations
   - Profils de pression énergétique

3. **S6 (Ablation) :**
   - Boxplots ODI par condition
   - Scatter plots (ODI vs PDI)
   - Distributions et corrélations

### Exemples d'utilisation des graphiques

```bash
# Générer tous les graphiques
python3 analyze_results.py outputs/test_s3/ --plot

# Les graphiques sont sauvés dans:
# outputs/test_s3/analysis_plots/summary_energy.png
```

## 🔍 Interprétation Avancée

### Métriques Clés

**ODI (Offloading Dependency Index) :**
- **Formule :** `ODI = Rappel_avec_aide - Rappel_sans_aide`
- **Interprétation :**
  - `ODI > 0.1` : Forte dépendance
  - `0 < ODI < 0.1` : Dépendance modérée  
  - `ODI ≈ 0` : Autonomie préservée
  - `ODI < 0` : Effet délétère de l'aide

**PDI (Performance Degradation Index) :**
- Mesure la dégradation structurelle
- Corrélé avec la consommation énergétique

**Indices Énergétiques :**
- `E_total` : Consommation totale
- `P_E` : Pression énergétique (dépassements)
- `E_trend` : Tendance temporelle

### Patterns Typiques

1. **Apprentissage Normal (S1-S2) :**
   - Activité stable ~8-12 Hz
   - Élagage modéré (PRUNE/GROW ≈ 1)

2. **Fenêtres Critiques (S3) :**
   - Pics énergétiques périodiques
   - Corrélation gamma-énergie

3. **Déchargement Efficace (S4) :**
   - ODI faible (~0.05)
   - Réduction énergétique post-t0

4. **Effets de Modulation (S6) :**
   - Relation linéaire Ω-ODI
   - Seuils de saturation

## 🚀 Workflow d'Analyse Recommandé

### 1. Exploration Initiale

```bash
# Vue d'ensemble des répertoires
ls outputs/ test1/

# Analyse rapide S2 (structure)
python3 analyze_results.py outputs/test_s2/
```

### 2. Analyse Énergétique

```bash
# S3 avec graphiques
python3 analyze_results.py outputs/test_s3/ --plot

# Examiner les pics énergétiques
# Vérifier les corrélations temporal
```

### 3. Évaluation du Déchargement

```bash
# S4 avec métriques ODI
python3 analyze_results.py test1/s4_results/ --plot

# Interpréter les indices de dépendance
# Corréler avec la consommation énergétique
```

### 4. Études Systématiques

```bash
# S6 pour les effets de modulation
python3 analyze_results.py test1/s6_ablation/ --plot

# Analyser les rapports HTML générés
# Vérifier la significativité statistique
```

### 5. Comparaisons Croisées

```bash
# Comparer plusieurs conditions
python3 analyze_results.py outputs/test_s2/ --plot
python3 analyze_results.py outputs/test_s3/ --plot

# Examiner l'évolution S2→S3→S4
```

## 📚 Ressources Additionnelles

### Fichiers de Configuration

Les paramètres utilisés sont sauvés dans `config_effective.json` :

```json
{
  "duration_ms": 8000,
  "offloading": {
    "t0_ms": 4000,
    "omega": 0.5
  },
  "energy": {
    "budget_B": 10000,
    "window_ms": 50
  }
}
```

### Scripts Spécialisés

- `src/run_s*.py` : Exécution des modèles
- `src/metrics/` : Calculs de métriques avancées  
- `src/stats/stats.py` : Tests statistiques
- `src/report/report_s6.py` : Génération de rapports

### Dépannage

**Erreur "KeyError" :** Structure de données différente
- Vérifiez les colonnes avec `pandas.read_csv(...).columns`
- Le script s'adapte automatiquement dans la plupart des cas

**Graphiques vides :** Données manquantes
- Vérifiez que les fichiers existent
- Examinez les logs pour les erreurs de chargement

**Tests statistiques échouent :** Données insuffisantes
- S6 nécessite plusieurs conditions (n>2)
- Vérifiez le taux de réussite des simulations

## 💡 Conseils d'Analyse

1. **Commencez simple** : Analysez d'abord S2, puis progressez vers S4-S6
2. **Utilisez les graphiques** : Les patterns visuels révèlent beaucoup
3. **Comparez les conditions** : L'analyse relative est plus informative
4. **Vérifiez la cohérence** : Les métriques doivent être corrélées logiquement
5. **Documentez vos observations** : Gardez trace des patterns intéressants

---

🔬 **Happy analyzing!** Ce modèle révèle des mécanismes fascinants de plasticité et d'adaptation cognitive. 