# Makefile pour neuro_offload_model
# Usage: make <target>

.PHONY: help install precommit fmt lint test clean docs docker s1 s2 s3 s4 s5 s6 pipeline smoke integration

# Configuration par défaut
PYTHON := python3
PIP := pip
SEED := 42
DURATION_S1 := 5000
DURATION_S2 := 8000
DURATION_S3 := 10000
DURATION_S4 := 12000

# Couleurs pour output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help: ## 📚 Afficher cette aide
	@echo "$(BLUE)=== NEURO_OFFLOAD_MODEL MAKEFILE ===$(NC)"
	@echo "$(YELLOW)Raccourcis pour développement et exécution$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# === SETUP ET ENVIRONNEMENT ===

install: ## 🔧 Installation complète avec dépendances de développement
	@echo "$(BLUE)Installation des dépendances...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .[dev,viz]
	@echo "$(GREEN)✅ Installation terminée$(NC)"

install-minimal: ## 🔧 Installation minimale (core seulement)
	$(PIP) install -e .

install-viz: ## 🎨 Installation avec dépendances visualisation
	$(PIP) install -e .[viz]

install-all: ## 🔧 Installation complète (toutes les dépendances optionnelles)
	$(PIP) install -e .[all]

precommit: ## 🔒 Configuration des hooks pre-commit
	@echo "$(BLUE)Configuration pre-commit...$(NC)"
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "$(GREEN)✅ Pre-commit configuré$(NC)"

# === QUALITÉ DE CODE ===

fmt: ## 🎨 Formatage automatique du code (black + isort)
	@echo "$(BLUE)Formatage du code...$(NC)"
	black src tests
	isort src tests
	@echo "$(GREEN)✅ Formatage terminé$(NC)"

lint: ## 🔍 Vérification statique du code
	@echo "$(BLUE)Vérification du code...$(NC)"
	flake8 src tests
	mypy src/ || true
	@echo "$(GREEN)✅ Lint terminé$(NC)"

check: fmt lint ## 🔍 Formatage + lint

# === TESTS ===

test: ## 🧪 Exécution des tests unitaires
	@echo "$(BLUE)Exécution des tests...$(NC)"
	pytest tests/ -v --cov=src --cov-report=term-missing

test-fast: ## ⚡ Tests rapides (sans couverture)
	pytest tests/ -q --disable-warnings

test-slow: ## 🐌 Tests lents et d'intégration
	pytest tests/ -v --cov=src -m "slow or integration"

test-coverage: ## 📊 Rapport de couverture détaillé
	pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)📁 Rapport HTML: htmlcov/index.html$(NC)"

# === MODÈLES S-1→S-6 ===

s1: ## 🧬 Exécution S-1 (modèle de base LIF+STDP)
	@echo "$(BLUE)🧬 Lancement S-1...$(NC)"
	$(PYTHON) -m src.run_s1 --config configs/s1.json --duration_ms $(DURATION_S1) --seed $(SEED) --plot --outdir outputs/s1_$(SEED)
	@echo "$(GREEN)✅ S-1 terminé$(NC)"

s2: ## 🔗 Exécution S-2 (plasticité structurelle)
	@echo "$(BLUE)🔗 Lancement S-2...$(NC)"
	$(PYTHON) -m src.run_s2 --config configs/s2.json --duration_ms $(DURATION_S2) --seed $(SEED) --plot --outdir outputs/s2_$(SEED)
	@echo "$(GREEN)✅ S-2 terminé$(NC)"

s3: ## 🎯 Exécution S-3 (fenêtres critiques + énergie)
	@echo "$(BLUE)🎯 Lancement S-3...$(NC)"
	$(PYTHON) -m src.run_s3 --config configs/s3.json --duration_ms $(DURATION_S3) --seed $(SEED) --plot --outdir outputs/s3_$(SEED)
	@echo "$(GREEN)✅ S-3 terminé$(NC)"

s4: ## 🤖 Exécution S-4 (offloading cognitif)
	@echo "$(BLUE)🤖 Lancement S-4...$(NC)"
	$(PYTHON) -m src.run_s4 --config configs/s4.json --duration_ms $(DURATION_S4) --seed $(SEED) --plot --outdir outputs/s4_$(SEED)
	@echo "$(GREEN)✅ S-4 terminé$(NC)"

s5: ## 📊 Exécution S-5 (connectivité EEG-like)
	@echo "$(BLUE)📊 Lancement S-5...$(NC)"
	$(PYTHON) -m src.run_s5 --config configs/s5.json --groups 20 --plot --seed $(SEED) --outdir outputs/s5_$(SEED)
	@echo "$(GREEN)✅ S-5 terminé$(NC)"

s6: ## 📈 Exécution S-6 (études d'ablation)
	@echo "$(BLUE)📈 Lancement S-6...$(NC)"
	$(PYTHON) -m src.run_s6 --config configs/s6.json --outdir outputs/s6_$(SEED) --mode run --seed $(SEED) --plot
	@echo "$(GREEN)✅ S-6 terminé$(NC)"

# === PIPELINES ===

smoke: ## 💨 Tests smoke rapides (configs minimales)
	@echo "$(BLUE)💨 Tests smoke...$(NC)"
	$(PYTHON) -c "from src.model_s1 import build_network_s1; config={'seed':1,'duration_ms':1000,'dt_ms':0.1,'N_e':50,'N_i':10,'p_connect_EI':0.1,'lif':{'tau_m_ms':20,'E_L_mV':-70,'V_th_mV':-50,'V_reset_mV':-60,'refractory_ms':5},'syn':{'gmax_e':0.6,'gmax_i':6.0,'w_init_mean':0.5,'w_init_std':0.1,'w_max':1.0,'tau_e_ms':5,'tau_i_ms':10},'stdp':{'tau_pre_ms':20,'tau_post_ms':20,'Apre':0.01,'Apost':-0.012},'scaling':{'enabled':True,'target_hz':5.0,'interval_ms':500,'eta_scale':0.01,'min_scale':0.5,'max_scale':2.0}}; net,monitors=build_network_s1(config); print('✅ S-1 smoke OK')"
	$(PYTHON) -c "from src.experiments.ablation_s6 import GridRunner; config={'omega':[0.0],'t0_ms':[1000],'seeds':[1],'replications':1}; runner=GridRunner(config); print('✅ S-6 smoke OK')"
	@echo "$(GREEN)✅ Tests smoke terminés$(NC)"

pipeline: s1 s2 s3 s4 s5 s6 ## 🚀 Pipeline complet S-1→S-6
	@echo "$(GREEN)🎉 Pipeline S-1→S-6 terminé avec succès!$(NC)"

integration: ## 🔄 Test d'intégration end-to-end
	@echo "$(BLUE)🔄 Test d'intégration...$(NC)"
	@mkdir -p outputs/integration_test
	@$(MAKE) smoke
	@$(MAKE) test-fast
	@echo "$(GREEN)✅ Intégration réussie$(NC)"

# === DOCKER ===

docker-build: ## 🐳 Construction de l'image Docker
	@echo "$(BLUE)🐳 Construction Docker...$(NC)"
	docker build -t neuro_offload_model:latest .
	@echo "$(GREEN)✅ Image Docker construite$(NC)"

docker-run: ## 🐳 Exécution du container Docker
	docker run --rm -it -v $(PWD)/outputs:/home/neuro/outputs -v $(PWD)/configs:/home/neuro/configs neuro_offload_model:latest

docker-test: ## 🐳 Tests dans le container Docker
	docker run --rm neuro_offload_model:latest

docker-shell: ## 🐳 Shell interactif dans le container
	docker run --rm -it -v $(PWD)/outputs:/home/neuro/outputs -v $(PWD)/configs:/home/neuro/configs neuro_offload_model:latest bash

docker-help: ## 🐳 Aide pour l'utilisation Docker
	docker run --rm neuro_offload_model:latest /home/neuro/docker_help.sh

# === DOCUMENTATION ===

docs: ## 📚 Construction de la documentation MkDocs
	@echo "$(BLUE)📚 Construction docs...$(NC)"
	mkdocs build
	@echo "$(GREEN)✅ Documentation construite dans site/$(NC)"

docs-serve: ## 📚 Serveur de documentation local
	@echo "$(BLUE)📚 Serveur docs sur http://localhost:8000$(NC)"
	mkdocs serve

docs-deploy: ## 📚 Déploiement de la documentation
	mkdocs gh-deploy

# === NETTOYAGE ===

clean: ## 🧹 Nettoyage des fichiers temporaires
	@echo "$(BLUE)🧹 Nettoyage...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	rm -rf outputs/test_* outputs/ci_*
	@echo "$(GREEN)✅ Nettoyage terminé$(NC)"

clean-all: clean ## 🧹 Nettoyage complet (inclut outputs/)
	rm -rf outputs/ site/
	@echo "$(GREEN)✅ Nettoyage complet terminé$(NC)"

# === RELEASE ===

version: ## 📋 Afficher la version actuelle
	@$(PYTHON) -c "import toml; print('Version:', toml.load('pyproject.toml')['project']['version'])"

check-release: ## ✅ Vérifications avant release
	@echo "$(BLUE)✅ Vérifications pré-release...$(NC)"
	@$(MAKE) lint
	@$(MAKE) test
	@$(MAKE) smoke
	@$(MAKE) docker-build
	@$(MAKE) docker-test
	@echo "$(GREEN)✅ Prêt pour release$(NC)"

# === DÉVELOPPEMENT ===

dev-setup: install precommit ## 🛠️ Setup complet pour développement
	@echo "$(GREEN)🎉 Environnement de développement prêt!$(NC)"
	@echo "$(YELLOW)Commandes utiles:$(NC)"
	@echo "  make s1     # Test S-1"
	@echo "  make test   # Tests unitaires"
	@echo "  make fmt    # Formatage code"
	@echo "  make docs-serve # Documentation"

# Configuration par défaut
.DEFAULT_GOAL := help 