#!/usr/bin/env bash
# Script d'exécution end-to-end pour neuro_offload_model
# Usage: ./scripts/run_end_to_end.sh [SEED]

set -euo pipefail

# Configuration
SEED=${1:-42}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/pipeline_${TIMESTAMP}_seed${SEED}"

# Couleurs pour output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_color() {
    local color=$1
    shift
    echo -e "${color}$*${NC}"
}

echo_color $BLUE "🚀 PIPELINE END-TO-END neuro_offload_model"
echo_color $BLUE "================================================"
echo_color $YELLOW "Seed: $SEED"
echo_color $YELLOW "Output: $OUTPUT_DIR"
echo_color $YELLOW "Timestamp: $TIMESTAMP"

# Vérification de l'environnement
echo_color $BLUE "\n🔍 Vérification de l'environnement..."

if ! command -v python3 &> /dev/null; then
    echo_color $RED "❌ Python3 non trouvé"
    exit 1
fi

if ! python3 -c "import brian2" 2>/dev/null; then
    echo_color $RED "❌ Brian2 non installé"
    echo_color $YELLOW "Installation: pip install -e .[dev,viz]"
    exit 1
fi

echo_color $GREEN "✅ Environnement OK"

# Création du répertoire de sortie
mkdir -p "$OUTPUT_DIR"

# Fonction d'exécution avec gestion d'erreurs
run_model() {
    local model=$1
    local config=$2
    local duration=$3
    local description=$4
    
    echo_color $BLUE "\n🧬 $description..."
    local start_time=$(date +%s)
    
    if python3 -m "src.run_$model" \
        --config "$config" \
        --duration_ms "$duration" \
        --seed "$SEED" \
        --plot \
        --outdir "$OUTPUT_DIR/$model"; then
        
        local end_time=$(date +%s)
        local duration_sec=$((end_time - start_time))
        echo_color $GREEN "✅ $model terminé en ${duration_sec}s"
        
        # Vérification des outputs
        if [[ -d "$OUTPUT_DIR/$model" ]]; then
            local png_count=$(find "$OUTPUT_DIR/$model" -name "*.png" | wc -l)
            local csv_count=$(find "$OUTPUT_DIR/$model" -name "*.csv" | wc -l)
            echo_color $GREEN "   📊 $png_count PNG, $csv_count CSV générés"
        fi
    else
        echo_color $RED "❌ $model échoué"
        return 1
    fi
}

# Pipeline principal
echo_color $BLUE "\n🎯 EXÉCUTION PIPELINE S-1→S-6"

# S-1: Modèle de base
run_model "s1" "configs/s1.json" "5000" "S-1: Modèle de base LIF+STDP"

# S-2: Plasticité structurelle  
run_model "s2" "configs/s2.json" "8000" "S-2: Plasticité structurelle"

# S-3: Fenêtres critiques
run_model "s3" "configs/s3.json" "10000" "S-3: Fenêtres critiques + énergie"

# S-4: Offloading cognitif
run_model "s4" "configs/s4.json" "12000" "S-4: Offloading cognitif"

# S-5: Connectivité EEG-like
echo_color $BLUE "\n📊 S-5: Connectivité EEG-like..."
if python3 -m src.run_s5 \
    --config configs/s5.json \
    --groups 20 \
    --plot \
    --seed "$SEED" \
    --outdir "$OUTPUT_DIR/s5"; then
    echo_color $GREEN "✅ S-5 terminé"
else
    echo_color $YELLOW "⚠️ S-5 échoué (non-critique)"
fi

# S-6: Études d'ablation (version accélérée)
echo_color $BLUE "\n📈 S-6: Études d'ablation (version rapide)..."
if python3 -c "
from src.experiments.ablation_s6 import GridRunner
from src.report.report_s6 import make_tables, plots, build_html_report
import os

# Configuration rapide pour end-to-end
config = {
    'omega': [0.0, 0.5],
    't0_ms': [3000], 
    'seeds': [$SEED],
    'replications': 1,
    's4_base_config': 'configs/s4.json'
}

outdir = '$OUTPUT_DIR/s6'
os.makedirs(outdir, exist_ok=True)

runner = GridRunner(config)
df = runner.run_grid(outdir, mode='direct')

# Génération du rapport
make_tables(df, outdir)
plots(df, outdir)
build_html_report(df, outdir)

print(f'✅ S-6 terminé: {len(df)} conditions')
"; then
    echo_color $GREEN "✅ S-6 terminé"
else
    echo_color $YELLOW "⚠️ S-6 échoué (non-critique)"
fi

# Résumé final
echo_color $BLUE "\n📋 RÉSUMÉ PIPELINE"
echo_color $BLUE "=================="

total_png=$(find "$OUTPUT_DIR" -name "*.png" | wc -l)
total_csv=$(find "$OUTPUT_DIR" -name "*.csv" | wc -l)
total_html=$(find "$OUTPUT_DIR" -name "*.html" | wc -l)
total_size=$(du -sh "$OUTPUT_DIR" | cut -f1)

echo_color $GREEN "✅ Pipeline terminé avec succès!"
echo_color $YELLOW "📁 Répertoire: $OUTPUT_DIR"
echo_color $YELLOW "📊 Fichiers générés:"
echo_color $YELLOW "   - $total_png images PNG"
echo_color $YELLOW "   - $total_csv fichiers CSV" 
echo_color $YELLOW "   - $total_html rapports HTML"
echo_color $YELLOW "💾 Taille totale: $total_size"

# Génération d'un index HTML
cat > "$OUTPUT_DIR/index.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Pipeline neuro_offload_model - $TIMESTAMP</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 10px; }
        .model { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .success { border-left: 4px solid #4CAF50; }
        .warning { border-left: 4px solid #FF9800; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧬 Pipeline neuro_offload_model</h1>
        <p><strong>Exécution:</strong> $TIMESTAMP</p>
        <p><strong>Seed:</strong> $SEED</p>
        <p><strong>Fichiers:</strong> $total_png PNG, $total_csv CSV, $total_html HTML</p>
    </div>
    
    <div class="model success">
        <h2>S-1: Modèle de base</h2>
        <p>LIF + STDP + Scaling homéostatique</p>
        <p><a href="s1/">📁 Voir résultats S-1</a></p>
    </div>
    
    <div class="model success">
        <h2>S-2: Plasticité structurelle</h2>
        <p>Alive gating + activity-dependent pruning</p>
        <p><a href="s2/">📁 Voir résultats S-2</a></p>
    </div>
    
    <div class="model success">
        <h2>S-4: Offloading cognitif</h2>
        <p>Modulation Ω(t) + LC-NE + ODI</p>
        <p><a href="s4/">📁 Voir résultats S-4</a></p>
    </div>
    
    <div class="model success">
        <h2>S-6: Études d'ablation</h2>
        <p>Grille Ω×t0 + analyses statistiques</p>
        <p><a href="s6/report_s6.html">📊 Voir rapport S-6</a></p>
    </div>
    
    <footer style="margin-top: 40px; color: #666;">
        <p>Généré par neuro_offload_model pipeline end-to-end</p>
    </footer>
</body>
</html>
EOF

echo_color $GREEN "\n🌐 Index HTML généré: $OUTPUT_DIR/index.html"
echo_color $BLUE "\n🎉 Pipeline end-to-end terminé avec succès!"

# Ouverture optionnelle du navigateur (si disponible)
if command -v open &> /dev/null; then
    echo_color $YELLOW "💡 Ouvrir dans le navigateur? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        open "$OUTPUT_DIR/index.html"
    fi
fi 