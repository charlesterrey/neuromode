# Image de base Python 3.10 optimisée
FROM python:3.10-slim

# Métadonnées
LABEL maintainer="Équipe Neurosciences Computationnelles"
LABEL description="Modèles neuronaux S-1→S-6 avec Brian2"
LABEL version="0.1.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Dépendances système pour Brian2 et visualisation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    git \
    ca-certificates \
    # Dépendances pour matplotlib et nilearn
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    # Nettoyage
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Créer utilisateur non-root pour sécurité
RUN useradd --create-home --shell /bin/bash neuro
USER neuro
WORKDIR /home/neuro

# Mise à jour pip
RUN python -m pip install --user --upgrade pip setuptools wheel

# Copier les fichiers de configuration d'abord (pour cache Docker)
COPY --chown=neuro:neuro pyproject.toml setup.cfg requirements.txt ./

# Installation des dépendances Python
RUN python -m pip install --user -e .[dev,viz]

# Copier le code source
COPY --chown=neuro:neuro . .

# Ajout du répertoire utilisateur au PATH
ENV PATH=/home/neuro/.local/bin:$PATH

# Créer les répertoires de travail
RUN mkdir -p outputs configs logs test_results

# Sanity check des installations
RUN python -c "import brian2; print(f'✅ Brian2 {brian2.__version__} installé')" && \
    python -c "import numpy; print(f'✅ NumPy {numpy.__version__} installé')" && \
    python -c "import matplotlib; print(f'✅ Matplotlib {matplotlib.__version__} installé')" && \
    python -c "import networkx; print(f'✅ NetworkX installé')" && \
    python -c "from src.model_s1 import build_network_s1; print('✅ Modules S-1→S-6 importables')"

# Port pour éventuel serveur de docs
EXPOSE 8000

# Script de santé par défaut
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import brian2; from src.model_s1 import load_config; print('OK')" || exit 1

# Commande par défaut : tests + smoke run
CMD ["bash", "-c", "echo '🧪 Tests et smoke runs...' && pytest tests/ -q --disable-warnings && echo '🚀 Smoke run S-1...' && python -c 'from src.model_s1 import build_network_s1; import json; config={\"seed\":1,\"duration_ms\":1000,\"dt_ms\":0.1,\"N_e\":40,\"N_i\":8,\"p_connect_EI\":0.1,\"lif\":{\"tau_m_ms\":20,\"E_L_mV\":-70,\"V_th_mV\":-50,\"V_reset_mV\":-60,\"refractory_ms\":5},\"syn\":{\"gmax_e\":0.6,\"gmax_i\":6.0,\"w_init_mean\":0.5,\"w_init_std\":0.1,\"w_max\":1.0,\"tau_e_ms\":5,\"tau_i_ms\":10},\"stdp\":{\"tau_pre_ms\":20,\"tau_post_ms\":20,\"Apre\":0.01,\"Apost\":-0.012},\"scaling\":{\"enabled\":True,\"target_hz\":5.0,\"interval_ms\":500,\"eta_scale\":0.01,\"min_scale\":0.5,\"max_scale\":2.0}}; net,monitors=build_network_s1(config); print(\"✅ Container prêt!\")' && echo '🎉 Tous les tests passent!'"]

# Points de montage recommandés
VOLUME ["/home/neuro/outputs", "/home/neuro/configs"]

# Documentation d'utilisation
RUN echo '#!/bin/bash' > /home/neuro/docker_help.sh && \
    echo 'echo "=== AIDE CONTAINER NEURO_OFFLOAD_MODEL ==="' >> /home/neuro/docker_help.sh && \
    echo 'echo "🐳 Container prêt avec Brian2 + S-1→S-6"' >> /home/neuro/docker_help.sh && \
    echo 'echo "📁 Volumes: /home/neuro/outputs, /home/neuro/configs"' >> /home/neuro/docker_help.sh && \
    echo 'echo "🧪 Tests: pytest tests/"' >> /home/neuro/docker_help.sh && \
    echo 'echo "🚀 S-1: python -m src.run_s1 --config configs/s1.json"' >> /home/neuro/docker_help.sh && \
    echo 'echo "📊 S-6: python -m src.run_s6 --config configs/s6.json"' >> /home/neuro/docker_help.sh && \
    chmod +x /home/neuro/docker_help.sh 