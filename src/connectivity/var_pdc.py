"""
Module d'analyse de connectivité directionnelle VAR + PDC/dDTF.

Implémente l'ajustement de modèles VAR multivariés et le calcul
des mesures de connectivité directionnelle (PDC, dDTF).
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import signal, linalg
import warnings


def select_var_order(X: np.ndarray, p_min: int = 1, p_max: int = 20, 
                     criterion: str = 'aic') -> int:
    """
    Sélection automatique de l'ordre VAR par critères d'information.
    
    Args:
        X: Signaux multivariés (T, K)
        p_min: Ordre minimal à tester
        p_max: Ordre maximal à tester
        criterion: Critère ('aic', 'bic')
        
    Returns:
        Ordre optimal p_opt
    """
    T, K = X.shape
    
    if p_max >= T // 4:
        p_max = max(1, T // 4 - 1)  # Sécurité pour éviter overfitting
    
    criteria_values = []
    
    for p in range(p_min, p_max + 1):
        try:
            A, Sigma = fit_var_ols(X, p, ridge_lambda=1e-6)
            
            # Calcul des résidus
            residuals = compute_var_residuals(X, A)
            
            # Log-vraisemblance (approximation gaussienne)
            log_likelihood = -0.5 * T * (K * np.log(2 * np.pi) + np.log(np.linalg.det(Sigma)))
            log_likelihood -= 0.5 * np.trace(residuals.T @ residuals @ np.linalg.inv(Sigma))
            
            # Critères d'information
            n_params = K * K * p  # Nombre de paramètres dans le modèle VAR(p)
            
            if criterion == 'aic':
                ic_value = -2 * log_likelihood + 2 * n_params
            elif criterion == 'bic':
                ic_value = -2 * log_likelihood + np.log(T) * n_params
            else:
                raise ValueError(f"Critère non supporté: {criterion}")
            
            criteria_values.append(ic_value)
            
        except (np.linalg.LinAlgError, ValueError):
            # En cas d'erreur numérique, pénaliser fortement
            criteria_values.append(np.inf)
    
    # Sélection du minimum
    if len(criteria_values) == 0 or all(np.isinf(criteria_values)):
        return p_min  # Fallback
    
    optimal_idx = np.argmin(criteria_values)
    p_opt = p_min + optimal_idx
    
    return p_opt


def fit_var_ols(X: np.ndarray, p: int, ridge_lambda: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ajustement d'un modèle VAR(p) par moindres carrés ordinaires + ridge.
    
    Modèle: X(t) = A₁X(t-1) + A₂X(t-2) + ... + AₚX(t-p) + ε(t)
    
    Args:
        X: Signaux (T, K)
        p: Ordre du modèle VAR
        ridge_lambda: Régularisation ridge
        
    Returns:
        A: Coefficients VAR (K, K, p)
        Sigma: Matrice de covariance du bruit (K, K)
    """
    T, K = X.shape
    
    if p >= T:
        raise ValueError(f"Ordre VAR p={p} trop grand pour T={T}")
    
    # Construction des matrices de régression
    Y = X[p:, :]  # Variables dépendantes (T-p, K)
    
    # Variables indépendantes: [X(t-1), X(t-2), ..., X(t-p)]
    X_lag = np.zeros((T - p, K * p))
    for lag in range(1, p + 1):
        start_col = (lag - 1) * K
        end_col = lag * K
        X_lag[:, start_col:end_col] = X[p - lag:-lag, :]
    
    # Régression ridge: (X'X + λI)⁻¹X'Y
    XtX = X_lag.T @ X_lag
    XtY = X_lag.T @ Y
    
    # Ajout de la régularisation ridge
    reg_matrix = ridge_lambda * np.eye(XtX.shape[0])
    
    try:
        # Résolution du système
        A_flat = linalg.solve(XtX + reg_matrix, XtY)  # (K*p, K)
        
        # Reshape en tenseur (K, K, p)
        A = np.zeros((K, K, p))
        for lag in range(p):
            start_row = lag * K
            end_row = (lag + 1) * K
            A[:, :, lag] = A_flat[start_row:end_row, :].T
        
        # Calcul des résidus et covariance
        Y_pred = X_lag @ A_flat
        residuals = Y - Y_pred
        
        # Covariance du bruit
        Sigma = (residuals.T @ residuals) / (T - p - K * p)
        
        # Régularisation de Sigma pour éviter singularité
        Sigma += ridge_lambda * np.eye(K)
        
        return A, Sigma
        
    except np.linalg.LinAlgError:
        # En cas d'erreur, retourner modèle trivial
        warnings.warn("Échec ajustement VAR, retour modèle trivial")
        A = np.zeros((K, K, p))
        Sigma = np.eye(K)
        return A, Sigma


def compute_var_residuals(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Calcule les résidus d'un modèle VAR ajusté.
    
    Args:
        X: Signaux observés (T, K)
        A: Coefficients VAR (K, K, p)
        
    Returns:
        Résidus (T-p, K)
    """
    T, K = X.shape
    _, _, p = A.shape
    
    Y = X[p:, :]  # Observations à prédire
    
    # Prédiction VAR
    Y_pred = np.zeros_like(Y)
    for t in range(T - p):
        for lag in range(1, p + 1):
            Y_pred[t, :] += A[:, :, lag - 1] @ X[p - lag + t, :]
    
    residuals = Y - Y_pred
    return residuals


def is_stable(A: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Vérifie la stabilité d'un modèle VAR.
    
    Un modèle VAR(p) est stable si toutes les racines du polynôme
    caractéristique det(I - A₁z - A₂z² - ... - Aₚzᵖ) = 0
    sont à l'extérieur du cercle unité.
    
    Args:
        A: Coefficients VAR (K, K, p)
        tolerance: Tolérance pour les valeurs proches de 1
        
    Returns:
        True si le modèle est stable
    """
    K, _, p = A.shape
    
    if p == 0:
        return True
    
    # Construction de la matrice compagnon
    # [A₁ A₂ ... Aₚ₋₁ Aₚ]
    # [I  0  ... 0    0 ]
    # [0  I  ... 0    0 ]
    # [0  0  ... I    0 ]
    
    companion = np.zeros((K * p, K * p))
    
    # Première ligne: coefficients VAR
    for lag in range(p):
        companion[:K, lag * K:(lag + 1) * K] = A[:, :, lag]
    
    # Lignes suivantes: matrices identité décalées
    for i in range(1, p):
        start_row = i * K
        end_row = (i + 1) * K
        start_col = (i - 1) * K
        end_col = i * K
        companion[start_row:end_row, start_col:end_col] = np.eye(K)
    
    # Calcul des valeurs propres
    try:
        eigenvalues = np.linalg.eigvals(companion)
        moduli = np.abs(eigenvalues)
        
        # Vérification: tous les modules < 1 (avec tolérance)
        return np.all(moduli < 1.0 - tolerance)
        
    except np.linalg.LinAlgError:
        return False


def freq_response(A: np.ndarray, fs: float, freqs: np.ndarray) -> np.ndarray:
    """
    Calcule la réponse fréquentielle A(f) du modèle VAR.
    
    A(f) = I - Σₚ Aₚ exp(-j2πfp/fs)
    
    Args:
        A: Coefficients VAR (K, K, p)
        fs: Fréquence d'échantillonnage
        freqs: Fréquences d'évaluation
        
    Returns:
        A_f: Réponse fréquentielle (len(freqs), K, K) complexe
    """
    K, _, p = A.shape
    n_freqs = len(freqs)
    
    A_f = np.zeros((n_freqs, K, K), dtype=complex)
    
    for f_idx, f in enumerate(freqs):
        A_f[f_idx, :, :] = np.eye(K)  # Terme I
        
        # Soustraction des termes Aₚ exp(-j2πfp/fs)
        for lag in range(1, p + 1):
            phase = -2j * np.pi * f * lag / fs
            A_f[f_idx, :, :] -= A[:, :, lag - 1] * np.exp(phase)
    
    return A_f


def pdc(A_f: np.ndarray) -> np.ndarray:
    """
    Calcule la Partial Directed Coherence (PDC).
    
    PDC_{ij}(f) = |A_{ij}(f)| / sqrt(Σₖ |A_{kj}(f)|²)
    
    Args:
        A_f: Réponse fréquentielle (n_freqs, K, K)
        
    Returns:
        PDC matrix (n_freqs, K, K) réelle
    """
    n_freqs, K, _ = A_f.shape
    PDC = np.zeros((n_freqs, K, K))
    
    for f_idx in range(n_freqs):
        A_f_current = A_f[f_idx, :, :]
        
        for j in range(K):
            # Normalisation par colonne j
            column_power = np.sum(np.abs(A_f_current[:, j]) ** 2)
            
            if column_power > 1e-12:  # Éviter division par zéro
                PDC[f_idx, :, j] = np.abs(A_f_current[:, j]) / np.sqrt(column_power)
    
    # Mettre la diagonale à zéro (pas d'auto-influence)
    for f_idx in range(n_freqs):
        np.fill_diagonal(PDC[f_idx, :, :], 0.0)
    
    return PDC


def ddtf(A_f: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    Calcule la Directed Transfer Function (dDTF).
    
    dDTF_{ij}(f) = |H_{ij}(f)|² σⱼ / Σₖ |H_{ik}(f)|² σₖ
    où H(f) = A(f)⁻¹
    
    Args:
        A_f: Réponse fréquentielle (n_freqs, K, K)
        Sigma: Matrice de covariance du bruit (K, K)
        
    Returns:
        dDTF matrix (n_freqs, K, K) réelle
    """
    n_freqs, K, _ = A_f.shape
    dDTF = np.zeros((n_freqs, K, K))
    
    # Extraction des variances du bruit (diagonale de Sigma)
    sigma_diag = np.diag(Sigma)
    
    for f_idx in range(n_freqs):
        try:
            # Calcul de H(f) = A(f)⁻¹
            H_f = np.linalg.inv(A_f[f_idx, :, :])
            
            for i in range(K):
                # Normalisation par ligne i
                row_weighted_power = np.sum(np.abs(H_f[i, :]) ** 2 * sigma_diag)
                
                if row_weighted_power > 1e-12:
                    for j in range(K):
                        dDTF[f_idx, i, j] = (np.abs(H_f[i, j]) ** 2 * sigma_diag[j]) / row_weighted_power
                        
        except np.linalg.LinAlgError:
            # En cas de matrice singulière
            dDTF[f_idx, :, :] = 0.0
    
    # Mettre la diagonale à zéro
    for f_idx in range(n_freqs):
        np.fill_diagonal(dDTF[f_idx, :, :], 0.0)
    
    return dDTF


def band_average(conn_f: np.ndarray, freqs: np.ndarray, 
                 f_lo: float, f_hi: float) -> np.ndarray:
    """
    Moyenne d'une mesure de connectivité dans une bande fréquentielle.
    
    Args:
        conn_f: Connectivité fréquentielle (n_freqs, K, K)
        freqs: Vecteur des fréquences
        f_lo: Fréquence basse de la bande
        f_hi: Fréquence haute de la bande
        
    Returns:
        Connectivité moyennée (K, K)
    """
    # Sélection des indices dans la bande
    band_mask = (freqs >= f_lo) & (freqs <= f_hi)
    
    if not np.any(band_mask):
        # Aucune fréquence dans la bande
        _, K, _ = conn_f.shape
        return np.zeros((K, K))
    
    # Moyenne dans la bande
    conn_band = np.mean(conn_f[band_mask, :, :], axis=0)
    
    return conn_band


def compute_var_pipeline(X: np.ndarray, fs: float, 
                        bands: Dict[str, List[float]],
                        p: Optional[int] = None,
                        ridge_lambda: float = 1e-3) -> Dict[str, Any]:
    """
    Pipeline complet d'analyse VAR + PDC/dDTF.
    
    Args:
        X: Signaux multivariés (T, K)
        fs: Fréquence d'échantillonnage
        bands: Dictionnaire des bandes {'alpha': [8, 12], 'beta': [13, 30]}
        p: Ordre VAR (None pour sélection auto)
        ridge_lambda: Régularisation
        
    Returns:
        Résultats complets (coefficients, stabilité, PDC, dDTF par bande)
    """
    T, K = X.shape
    
    # Sélection de l'ordre VAR
    if p is None:
        p_opt = select_var_order(X, p_min=1, p_max=min(20, T // 4))
    else:
        p_opt = p
    
    # Ajustement du modèle VAR
    A, Sigma = fit_var_ols(X, p_opt, ridge_lambda)
    
    # Vérification de stabilité
    stable = is_stable(A)
    
    # Calcul des réponses fréquentielles
    freqs = np.linspace(0, fs/2, 256)  # Jusqu'à la fréquence de Nyquist
    A_f = freq_response(A, fs, freqs)
    
    # Calcul PDC et dDTF
    PDC = pdc(A_f)
    dDTF = ddtf(A_f, Sigma)
    
    # Moyennage par bandes
    results = {
        'p_opt': p_opt,
        'A': A,
        'Sigma': Sigma,
        'stable': stable,
        'freqs': freqs,
        'PDC_f': PDC,
        'dDTF_f': dDTF,
        'bands': {}
    }
    
    for band_name, (f_lo, f_hi) in bands.items():
        PDC_band = band_average(PDC, freqs, f_lo, f_hi)
        dDTF_band = band_average(dDTF, freqs, f_lo, f_hi)
        
        results['bands'][band_name] = {
            'PDC': PDC_band,
            'dDTF': dDTF_band,
            'f_range': [f_lo, f_hi]
        }
    
    return results 