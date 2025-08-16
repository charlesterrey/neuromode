# Modèles Neuronaux S-1 à S-6 : Développement Neural, Offloading et Analyses d'Ablation

## Description

Ce projet implémente des modèles neurobiologiques du développement cérébral, de l'offloading cognitif et des analyses de connectivité en six étapes :

### S-1 : Base LIF + STDP + Scaling Homéostatique

Le modèle S-1 établit la base avec :

- **Neurones LIF (Leaky Integrate-and-Fire)** avec conductances excitatrices et inhibitrices
- **Plasticité STDP (Spike-Timing Dependent Plasticity)** pair-based sur les connexions E→E
- **Scaling homéostatique** pour maintenir l'activité proche d'un niveau cible

### S-2 : Extension avec Plasticité Structurelle Développementale

Le modèle S-2 ajoute les mécanismes de remodelage structural :

- **Préallocation full E→E** : Toutes les connexions possibles (sauf auto-connexions) sont préallouées
- **Alive gating** : Variable binaire `alive` ∈ {0,1} contrôle l'activation des synapses
- **Phase GROW** : Surcroissance avec activation progressive des connexions jusqu'à densité cible
- **Phase PRUNE** : Élagage activité-dépendant selon la règle A + coût de câblage
- **Score d'activité A** : Accumulation locale de l'activité pré/post-synaptique pour l'élagage
- **Coûts de câblage** : Pénalisation des connexions longues (économie spatiale)
- **Règle d'élagage** : p_prune = σ(k₁(θ_act - A) + k₂·len_cost)

### S-3 : Extension avec Fenêtres Critiques γ(t) et Contraintes Énergétiques

Le modèle S-3 ajoute la modulation développementale et les pressions métaboliques :

- **Fenêtres critiques γ(t)** : Modulation temporelle de la plasticité STDP (Apre, Apost multiplié par γ(t))
- **Profiles γ(t)** : Double-sigmoïde (ouverture/fermeture) ou gaussien (pic de plasticité)
- **Budget énergétique** : Suivi des coûts spikes + synapses + câblage dans fenêtre glissante
- **Pression énergétique P_E** : P_E = max(0, (E_win - B)/B) accélère l'élagage quand budget dépassé
- **Règle d'élagage S-3** : p_prune = σ(γ·k₁(θ_act - A) + k₂·len_cost + kE·P_E)
- **Contraintes métaboliques** : E_win = c_spike·#spikes + c_syn·#syn_events + c_len·Σ(len_cost·alive)

### S-4 : Extension avec Offloading Cognitif Ω(t) et Contrôle LC-NE

Le modèle S-4 simule l'offloading mémoire et la modulation neuromodulatrice :

- **Variable d'offloading Ω(t)** : Transition sigmoïde après t₀ réduisant l'effort endogène
- **Contrôle LC-NE g_NE(t)** : Neuromodulation locus coeruleus-noradrénaline dépendante de l'effort
- **Modulation conjointe** : STDP modulé par γ(t) * g_NE(t) (effort↓ → g_NE↓ → plasticité↓)
- **Protocole de probe** : Tests rappel avec/sans assistance pour mesurer dépendance
- **Index ODI** : ODI = rappel_sans_aide - rappel_avec_aide (ODI>0 = dépendance à l'aide externe)
- **Entrées dépendantes effort** : Taux Poisson externes modulés par effort(t)

### S-5 : Analyse de Connectivité EEG-like (PDC/dDTF) + Visualisation 3D

Le modèle S-5 extrait la connectivité directionnelle depuis les patterns de spikes :

- **Signaux LFP proxy** : Conversion spikes → taux → regroupement → lissage alpha → signaux multi-canaux
- **Modèles VAR(p)** : Ajustement autorégressif multivarié avec sélection automatique d'ordre (AIC/BIC)
- **PDC (Partial Directed Coherence)** : Connectivité directionnelle normalisée par fréquence
- **dDTF (directed DTF)** : Transfer function pondérée par variance du bruit
- **Moyennage fréquentiel** : Extraction par bandes (alpha, bêta) pour comparaison neurophysiologique
- **Pipeline offline** : Analyse post-simulation pour études de connectivité développementale

### S-6 : Études d'Ablation Systématiques (Grille Ω×t0×seeds)

Le modèle S-6 orchestre des explorations paramétriques complètes :

- **Grille expérimentale** : Exploration systématique de l'espace Ω (offloading) × t0 (timing) × seeds (reproductibilité)
- **Métriques intégrées** : ODI (Offloading Dependency Index), PDI (Plasticity Dependency Index), énergie métabolique
- **Analyses statistiques** : ANOVA bidirectionnelle, tests de permutation, corrélations (Spearman)
- **Tables publication-ready** : Export CSV + LaTeX avec intervalles de confiance bootstrap
- **Visualisations** : Heatmaps, barplots avec CI, scatterplots, violonplots par condition
- **Rapports HTML** : Synthèse automatique avec tendances principales et note éthique

### Contexte Neurobiologique

Le projet s'appuie sur les mécanismes fondamentaux du développement neural :

- **Surproduction synaptique** : Formation initiale d'un grand nombre de connexions (S-2)
- **Plasticité Hebienne/STDP** : Renforcement des connexions corrélées dans le temps (S-1, S-2)
- **Élagage activité-dépendant** : Élimination des connexions peu utilisées ou coûteuses (S-2)
- **Régulation homéostatique** : Maintien de l'équilibre excitation/inhibition (S-1, S-2)
- **Contraintes spatiales** : Économie du câblage neuronal (S-2)

Les étapes futures (S-3/S-4) incluront la neuromodulation LC-NE et les contraintes énergétiques.

### Références Scientifiques

- Hensch, T.K. (2005). Critical period plasticity in local cortical circuits. *Nature Reviews Neuroscience*
- Huttenlocher, P.R. (2002). Neural plasticity: The effects of environment on the development of the cerebral cortex
- Turrigiano, G.G. (2008). The self-tuning neuron: synaptic scaling of excitatory synapses

## Installation

### Prérequis

- Python 3.7+
- Environnement virtuel recommandé

### Installation des dépendances

```bash
# Créer un environnement virtuel
python -m venv venv_neuro

# Activer l'environnement (Linux/Mac)
source venv_neuro/bin/activate

# Activer l'environnement (Windows)
# venv_neuro\\Scripts\\activate

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### Exécution S-1 (Base)

```bash
# Simulation S-1 avec configuration par défaut
python -m src.run_s1 --config configs/default.json --duration_ms 5000 --plot

# Simulation courte pour test
python -m src.run_s1 --config configs/default.json --duration_ms 1000 --plot --seed 42
```

### Exécution S-2 (Plasticité Structurelle)

```bash
# Simulation S-2 avec phases développementales
python -m src.run_s2 --config configs/s2.json --plot

# Simulation S-2 courte pour test
python -m src.run_s2 --config configs/s2.json --seed 42 --plot
```

### Options CLI

- `--config` : Chemin vers le fichier de configuration JSON
- `--duration_ms` : Durée de simulation en millisecondes
- `--outdir` : Répertoire de sortie personnalisé
- `--plot` : Génération automatique des graphiques
- `--seed` : Graine aléatoire pour reproductibilité

### Configuration

#### S-1 : Paramètres de base dans `configs/default.json` :

```json
{
  "N_e": 400,           // Nombre de neurones excitateurs
  "N_i": 100,           // Nombre de neurones inhibiteurs
  "p_connect": 0.1,     // Probabilité de connexion
  "scaling": {
    "enabled": true,
    "target_hz": 5.0,   // Taux cible pour le scaling
    "interval_ms": 500  // Intervalle d'application du scaling
  }
}
```

#### S-2 : Paramètres développementaux dans `configs/s2.json` :

```json
{
  "struct": {
    "phase": {
      "T_grow_ms": 3000,    // Durée phase GROW (surcroissance)
      "T_prune_ms": 5000    // Durée phase PRUNE (élagage)
    },
    "rho_target_grow": 0.25,  // Densité synaptique cible en phase GROW
    "theta_act": 0.15,        // Seuil d'activité pour élagage
    "k1": 8.0,                // Poids terme activité dans élagage
    "k2": 3.0,                // Poids coût câblage dans élagage
    "beta_pre": 0.02,         // Incrément A pour événement pré-synaptique
    "beta_post": 0.02         // Incrément A pour événement post-synaptique
  }
}
```

#### S-3 : Paramètres γ(t) et énergie dans `configs/s3.json` :

```json
{
  "gamma": {
    "mode": "double_sigmoid",  // Profil fenêtre critique
    "gamma_max": 1.0,          // Modulation maximale
    "t_open_ms": 1500,         // Début ouverture fenêtre
    "t_close_ms": 5500,        // Début fermeture fenêtre
    "slope_open": 0.01,        // Pente ouverture
    "slope_close": 0.01        // Pente fermeture
  },
  "energy": {
    "window_ms": 200,          // Fenêtre glissante énergétique
    "budget_B": 5000.0,        // Budget énergétique total
    "c_spike": 1.0,            // Coût par spike
    "c_syn": 0.1,              // Coût par événement synaptique
    "c_len": 0.01              // Coût par unité de câblage
  },
  "struct": {
    "kE": 5.0                  // Poids pression énergétique dans élagage
  }
}
```

## Sorties

### S-1 : Fichiers de base

- `spikes_e.csv` : Trains de spikes excitateurs (neurone_id, temps_ms)
- `spikes_i.csv` : Trains de spikes inhibiteurs
- `rates_e.csv` : Taux de décharge de la population excitatrice
- `rates_i.csv` : Taux de décharge de la population inhibitrice
- `weight_trajectories.csv` : Évolution temporelle des poids échantillonnés
- `final_weights_ee.npy` : Matrice finale des poids E→E
- `config_effective.json` : Configuration utilisée pour la simulation
- `network_stats.json` : Statistiques du réseau

### S-2 : Fichiers étendus (plasticité structurelle)

En plus des fichiers S-1 :

- `density_evolution.csv` : Évolution de la densité synaptique vs temps
- `structural_events.csv` : Log des événements GROW/PRUNE avec timestamps
- `activity_trajectories.npy` : Évolution des scores d'activité A
- `alive_trajectories.npy` : Évolution des états alive (0/1)
- `final_activity_ee.npy` : Scores d'activité finaux de toutes les synapses
- `final_alive_ee.npy` : États finaux alive de toutes les synapses
- `final_len_costs_ee.npy` : Coûts de câblage de toutes les synapses
- `degree_histogram.csv` : Distribution des degrés entrants/sortants
- `adjacency_t*.npy` : Instantanés périodiques de la matrice d'adjacence

### Graphiques (avec `--plot`)

#### S-1 :
- `raster_plot.png` : Raster des spikes E et I
- `population_rates.png` : Évolution des taux de décharge
- `weight_distributions.png` : Histogrammes des poids initiaux vs finaux
- `weight_trajectories.png` : Trajectoires temporelles des poids

#### S-2 :
- `density_evolution.png` : **Courbe clé** - Densité synaptique vs temps avec phases GROW/PRUNE
- `raster_plot.png` : Raster avec marqueurs de phases
- `population_rates.png` : Taux avec ligne de séparation des phases
- `degree_histogram.png` : Distribution des degrés entrants/sortants finaux
- `trajectories_sample.png` : Évolution des poids, activité A et états alive

## Interprétation des résultats S-2

### Courbe de densité synaptique (clé de l'analyse)

La courbe `density_evolution.png` doit montrer le pattern développemental attendu :

1. **Phase GROW (0 → T_grow)** : Augmentation progressive de la densité jusqu'à ~rho_target_grow
2. **Phase PRUNE (T_grow → T_total)** : Diminution de la densité par élagage sélectif

**Interprétation** :
- Pente positive en phase GROW → Surcroissance active
- Pente négative en phase PRUNE → Élagage activité-dépendant 
- Densité finale < densité maximale → Maturation réussie
- Plateau en phase PRUNE → Équilibre grow/prune atteint

### Règle d'élagage

Les synapses sont élaguées selon :
```
p_prune = σ(k₁(θ_act - A) + k₂·len_cost)
```

- **A < θ_act** : Synapse peu active → élagage probable
- **len_cost élevé** : Connexion longue → élagage probable  
- **k₁, k₂** : Balance activité vs économie spatiale

## Tests

```bash
# Tests S-1
python -m pytest tests/test_s1.py -v
python tests/test_s1.py

# Tests S-2
python -m pytest tests/test_s2.py -v  
python tests/test_s2.py
```

Les tests vérifient :
- ✅ Absence de NaN dans les poids
- ✅ Émission de spikes par le réseau
- ✅ Respect des bornes des poids [0, w_max]
- ✅ Génération correcte des fichiers de sortie
- ✅ Fonctionnement du scaling homéostatique

## Architecture

```
src/
├── model_s1.py          # Modèle principal (LIF + STDP + scaling)
├── run_s1.py            # Script d'exécution CLI
└── utils/
    └── io.py            # Utilitaires E/S et logging

configs/
└── default.json         # Configuration par défaut

tests/
└── test_s1.py          # Tests d'intégration

outputs/                 # Résultats horodatés (créé automatiquement)
figs/                   # Graphiques (créé automatiquement)
```

## Mécanismes Implémentés

### 1. Neurones LIF avec Conductances

```python
dv/dt = (E_L - v + g_e*(0mV - v) + g_i*(-80mV - v)) / tau_m
dg_e/dt = -g_e / tau_e
dg_i/dt = -g_i / tau_i
```

### 2. STDP Pair-Based

- **Potentiation** : Pré → Post (Δt > 0) avec A_pre = +0.01
- **Dépression** : Post → Pré (Δt < 0) avec A_post = -0.012
- **Bornes** : Poids clampés dans [0, w_max]

### 3. Scaling Homéostatique

Ajustement périodique des poids entrants pour maintenir le taux de décharge proche de `target_hz` :

```python
factor = 1 + eta * (target_rate / current_rate - 1)
w_new = clip(w_old * factor, min_scale * w_old, max_scale * w_old)
```

## Performance

- **Temps d'exécution** : ~10-30 secondes pour 5000 ms simulées (réseau 500 neurones)
- **Mémoire** : ~100-500 MB selon la taille du réseau
- **Optimisations** : Échantillonnage des poids, limitation des spikes enregistrés

## Extensions Futures (S-2/S-3)

### Étapes Planifiées

1. **S-2** : Ajout de la plasticité BCM et règles de pruning
2. **S-3** : Neuromodulation LC-NE pour fenêtres critiques
3. **S-4** : Contraintes énergétiques et coûts de câblage

### Points d'Extension

- `model_s1.py` → Nouvelles règles de plasticité dans `build_network()`
- `run_s1.py` → Paramètres additionnels dans la configuration
- Tests → Validation des nouveaux mécanismes

## Troubleshooting

### Problèmes Courants

**Erreur "ModuleNotFoundError: No module named 'brian2'"**
```bash
pip install brian2>=2.6
```

**Simulation trop lente**
- Réduire `N_e`, `N_i` ou `duration_ms`
- Augmenter `dt_ms` (attention à la stabilité numérique)

**Poids qui explosent**
- Réduire `Apre`, `Apost` ou `eta_scale`
- Vérifier les paramètres `gmax_e`, `gmax_i`

**Pas de spikes émis**
- Augmenter `gmax_e` ou réduire `V_th`
- Vérifier l'initialisation des potentiels `v`

## Licence

Modèle développé à des fins de recherche scientifique. Utilisation libre pour projets académiques. 