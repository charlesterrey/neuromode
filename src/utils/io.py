"""
Utilitaires d'entrée/sortie pour le modèle neuronal.
"""

import json
import os
import sys
from datetime import datetime
from typing import Any, Union
import numpy as np
import pandas as pd


def save_json(obj: Any, path: str) -> None:
    """Sauvegarde un objet en format JSON."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_array(name: str, arr: np.ndarray, outdir: str, format: str = 'npy') -> str:
    """Sauvegarde un array numpy en format NPY ou CSV."""
    ensure_dir(outdir)
    
    if format == 'npy':
        path = os.path.join(outdir, f"{name}.npy")
        np.save(path, arr)
    elif format == 'csv':
        path = os.path.join(outdir, f"{name}.csv")
        if arr.ndim == 1:
            pd.Series(arr).to_csv(path, index=False)
        else:
            pd.DataFrame(arr).to_csv(path, index=False)
    else:
        raise ValueError(f"Format non supporté: {format}")
    
    return path


def ensure_dir(path: str) -> None:
    """Crée le répertoire s'il n'existe pas."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def create_output_dir(base_dir: str = "outputs") -> str:
    """Crée un répertoire de sortie horodaté."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(base_dir, f"run_{timestamp}")
    ensure_dir(outdir)
    return outdir


def log(message: str, prefix: str = "[INFO]") -> None:
    """Logger simple avec préfixe et flush."""
    print(f"{prefix} {message}", flush=True)


def log_info(message: str) -> None:
    """Log d'information."""
    log(message, "[INFO]")


def log_warning(message: str) -> None:
    """Log d'avertissement."""
    log(message, "[WARN]")


def log_error(message: str) -> None:
    """Log d'erreur."""
    log(message, "[ERROR]") 