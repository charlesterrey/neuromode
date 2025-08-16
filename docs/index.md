# neuro_offload_model

## 🧬 Aperçu

**neuro_offload_model** est une suite complète de modèles neuronaux computationnels implémentant le développement cérébral, la plasticité structurelle, et l'offloading cognitif.

### Pipeline S-1→S-6

- **S-1** : Modèle de base LIF + STDP + Scaling homéostatique
- **S-2** : Plasticité structurelle avec alive gating et activity-dependent pruning  
- **S-3** : Fenêtres critiques γ(t) + budget énergétique métabolique
- **S-4** : Offloading cognitif (Ω, t0) + modulation LC-NE
- **S-5** : Analyse connectivité EEG-like (PDC/dDTF) + visualisation 3D
- **S-6** : Études d'ablation systématiques avec analyses statistiques

## 🚀 Démarrage rapide

### Installation

```bash
# Installation complète avec visualisation
pip install -e .[dev,viz]

# Configuration pre-commit pour qualité de code
make precommit

# Tests smoke
make smoke
```

### Utilisation de base

```bash
# Modèle de base S-1
make s1

# Plasticité structurelle S-2  
make s2

# Offloading cognitif S-4
make s4

# Études d'ablation S-6
make s6
```

## 📊 Fonctionnalités clés

### Modélisation neurobiologique
- **Brian2** : Simulation de réseaux de neurones LIF
- **STDP** : Plasticité synaptique spike-timing dependent
- **Scaling homéostatique** : Régulation des taux de décharge
- **Alive gating** : Gestion dynamique de la connectivité

### Plasticité structurelle
- **Overgrowth/Pruning** : Croissance puis élagage synaptique
- **Activity-dependent** : Pruning basé sur l'activité locale
- **Wiring cost** : Coûts de câblage par distance euclidienne
- **Energy budget** : Contraintes métaboliques réalistes

### Offloading cognitif
- **Ω(t)** : Variable d'offloading temporel
- **LC-NE** : Modulation neuromodulatrice (Locus Coeruleus)
- **ODI** : Offloading Dependency Index comportemental
- **Probe recall** : Tests de mémoire avec/sans assistance

### Analyses avancées
- **PDC/dDTF** : Connectivité directionnelle EEG-like
- **MVAR** : Modèles autorégressifs multivariés
- **Visualisation 3D** : Nilearn + Plotly pour rendu spatial
- **Tables publication-ready** : Export CSV + LaTeX automatique

## 🛠️ Développement

### Configuration environnement
```bash
# Setup complet développement
make dev-setup

# Formatage + lint
make check

# Tests complets
make test
```

### Docker
```bash
# Construction image
make docker-build

# Tests containerisés
make docker-test

# Shell interactif
make docker-shell
```

## 📚 Documentation détaillée

- [Guide d'installation](installation.md)
- [Tutoriel S-1→S-6](tutorial.md) 
- [Référence API](api.md)
- [Exemples d'utilisation](examples.md)
- [Guide contribution](contributing.md)

## 🔬 Applications scientifiques

### Neurosciences computationnelles
- Modélisation du développement cérébral
- Études de plasticité synaptique
- Simulation de fenêtres critiques
- Analyse de connectivité fonctionnelle

### Sciences cognitives  
- Modélisation de l'offloading cognitif
- Études de dépendance technologique
- Analyse coût-bénéfice métabolique
- Protocoles de rappel mnésique

### Applications cliniques (potentielles)
- Modélisation neurodéveloppementale
- Études de neuroplasticité thérapeutique
- Analyse d'efficacité énergétique cérébrale

## ⚠️ Avertissement éthique

Les résultats de ces modèles constituent des **hypothèses computationnelles** et non des preuves empiriques. Ils doivent être interprétés avec prudence dans le contexte de recherche en neurosciences cognitives.

## 📄 Licence

MIT License - Utilisation libre pour projets académiques et de recherche.

## 🤝 Contribution

Contributions bienvenues ! Voir [guide de contribution](contributing.md) pour les détails.

---

*Développé avec ❤️ pour la communauté des neurosciences computationnelles* 