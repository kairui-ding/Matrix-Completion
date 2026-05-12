#!/usr/bin/env python3
"""
Refined simulation / comparison code for the generalized matrix completion paper.

Model:
    M* = W A* Z^T + W B*^T + C* Z^T + L*

This file separates two different experimental goals:

1. Theorem study (`--study theorem`):
   - uses ONLY our one-stage estimator;
   - uses all observed entries for fitting, with no train/validation/test split;
   - uses the theorem-style fixed lambda
         lambda = C_lambda * sigma * sqrt(max(n1,n2) * p_fit);
   - reports component errors for A, B, C, L and infinity-type errors.

2. Method-comparison study (`--study comparison`):
   - uses train/validation/test split;
   - tunes lambda for each method using held-out NOISY validation observations;
   - reports clean recovery metrics in simulation, plus noisy held-out RMSE.

The theorem study should be used to check the theorem's predicted scaling.
The comparison study should be used for practical benchmarking and real-data-style evaluation.
"""

from __future__ import annotations

import argparse
import math
import os

# Keep linear algebra calls from oversubscribing threads in repeated SVD loops.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Basic utilities
# ============================================================

def fro_norm(X: np.ndarray) -> float:
    return float(np.linalg.norm(X, ord="fro"))


def spectral_norm(X: np.ndarray) -> float:
    return float(np.linalg.norm(X, ord=2))


def relative_fro_error(X_hat: np.ndarray, X_true: np.ndarray, eps: float = 1e-12) -> float:
    return fro_norm(X_hat - X_true) / (fro_norm(X_true) + eps)


def inf_error(X_hat: np.ndarray, X_true: np.ndarray) -> float:
    return float(np.max(np.abs(X_hat - X_true)))


def norm_2inf(X: np.ndarray) -> float:
    """Maximum row Euclidean norm."""
    if X.size == 0:
        return 0.0
    return float(np.max(np.linalg.norm(X, axis=1)))


def rmse_on_mask(X_hat: np.ndarray, X_target: np.ndarray, mask: np.ndarray) -> float:
    denom = float(np.sum(mask))
    if denom <= 0:
        return float("nan")
    err = mask * (X_hat - X_target)
    return math.sqrt(float(np.sum(err ** 2)) / denom)


def projector_onto_colspace(X: np.ndarray) -> np.ndarray:
    return X @ np.linalg.pinv(X.T @ X) @ X.T


def projector_onto_orthogonal_complement(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    return np.eye(n) - projector_onto_colspace(X)


def project_B_to_feasible(B: np.ndarray, PZ_perp: np.ndarray) -> np.ndarray:
    return PZ_perp @ B


def project_C_to_feasible(C: np.ndarray, PW_perp: np.ndarray) -> np.ndarray:
    return PW_perp @ C


def project_L_to_feasible(L: np.ndarray, PW_perp: np.ndarray, PZ_perp: np.ndarray) -> np.ndarray:
    return PW_perp @ L @ PZ_perp


def svt(Y: np.ndarray, tau: float) -> np.ndarray:
    U, s, Vh = np.linalg.svd(Y, full_matrices=False)
    s_new = np.maximum(s - tau, 0.0)
    return U @ np.diag(s_new) @ Vh


def numerical_rank(X: np.ndarray, rel_thresh: float = 1e-3) -> int:
    s = np.linalg.svd(X, compute_uv=False)
    if s.size == 0 or s[0] <= 1e-15:
        return 0
    return int(np.sum(s > rel_thresh * s[0]))


def recommended_lambda(sigma: float, n1: int, n2: int, p_fit: float, c_lambda: float = 5.0) -> float:
    """
    Theorem-style lambda scale for the one-stage nuclear-norm estimator.
    """
    return c_lambda * sigma * math.sqrt(max(n1, n2) * max(p_fit, 1e-12))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ============================================================
# Data generation
# ============================================================

def generate_truth(
    n1: int,
    n2: int,
    d1: int,
    d2: int,
    r_true: int,
    rng: np.random.Generator,
    scale_A: float = 1.0,
    scale_B: float = 1.0,
    scale_C: float = 1.0,
    scale_L: float = 1.0,
    normalize_covariates: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a feasible truth satisfying
        Z^T B = 0, W^T C = 0, W^T L = 0, L Z = 0.
    """
    W = rng.standard_normal((n1, d1))
    Z = rng.standard_normal((n2, d2))
    if normalize_covariates:
        W = W / math.sqrt(n1)
        Z = Z / math.sqrt(n2)

    PW_perp = projector_onto_orthogonal_complement(W)
    PZ_perp = projector_onto_orthogonal_complement(Z)

    A_true = scale_A * rng.standard_normal((d1, d2))
    B_true = project_B_to_feasible(scale_B * rng.standard_normal((n2, d1)), PZ_perp)
    C_true = project_C_to_feasible(scale_C * rng.standard_normal((n1, d2)), PW_perp)

    U0 = PW_perp @ rng.standard_normal((n1, r_true))
    V0 = PZ_perp @ rng.standard_normal((n2, r_true))
    U_true, _ = np.linalg.qr(U0)
    V_true, _ = np.linalg.qr(V0)
    sing_vals = np.linspace(2.0, 1.0, r_true)
    Sigma = np.diag(scale_L * sing_vals)
    L_true = U_true[:, :r_true] @ Sigma @ V_true[:, :r_true].T
    L_true = project_L_to_feasible(L_true, PW_perp, PZ_perp)

    M_true = W @ A_true @ Z.T + W @ B_true.T + C_true @ Z.T + L_true
    return W, Z, A_true, B_true, C_true, L_true, M_true


def generate_noisy_obs(
    M_true: np.ndarray,
    p_obs: float,
    sigma: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        Y_obs: observed noisy matrix, zeros on unobserved entries;
        mask_obs: {0,1} observation mask;
        Y_full_noisy: full noisy signal M_true + noise, useful only in simulation.
    """
    noise = rng.normal(0.0, sigma, size=M_true.shape)
    Y_full_noisy = M_true + noise
    mask_obs = (rng.random(M_true.shape) < p_obs).astype(float)
    Y_obs = mask_obs * Y_full_noisy
    return Y_obs, mask_obs, Y_full_noisy


def split_mask(
    mask_obs: np.ndarray,
    train_frac: float,
    val_frac: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split observed entries into train/validation/test masks.
    Fractions are relative to the observed entries.
    """
    if train_frac < 0 or val_frac < 0 or train_frac + val_frac > 1:
        raise ValueError("Need train_frac >= 0, val_frac >= 0, and train_frac + val_frac <= 1.")

    train = np.zeros_like(mask_obs)
    val = np.zeros_like(mask_obs)
    test = np.zeros_like(mask_obs)

    idx = np.argwhere(mask_obs > 0.5)
    perm = rng.permutation(len(idx))
    n_train = int(train_frac * len(idx))
    n_val = int(val_frac * len(idx))

    for k in perm[:n_train]:
        i, j = idx[k]
        train[i, j] = 1.0
    for k in perm[n_train:n_train + n_val]:
        i, j = idx[k]
        val[i, j] = 1.0
    for k in perm[n_train + n_val:]:
        i, j = idx[k]
        test[i, j] = 1.0
    return train, val, test


# ============================================================
# Estimators
# ============================================================

def fit_cov_interaction_only(
    Y: np.ndarray,
    mask: np.ndarray,
    W: np.ndarray,
    Z: np.ndarray,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Baseline nested in our model:
        M = W A Z^T + W B^T + C Z^T,
    with no latent low-rank L.

    This is not merely additive; it includes the interaction W A Z^T.
    """
    n1, n2 = Y.shape
    d1 = W.shape[1]
    d2 = Z.shape[1]
    p = max(float(np.mean(mask)), 1e-8)

    PW_perp = projector_onto_orthogonal_complement(W)
    PZ_perp = projector_onto_orthogonal_complement(Z)

    A = np.zeros((d1, d2))
    B = np.zeros((n2, d1))
    C = np.zeros((n1, d2))

    opW = spectral_norm(W.T @ W)
    opZ = spectral_norm(Z.T @ Z)
    etaA = 0.6 * p / (opW * opZ + 1e-12)
    etaB = 0.6 * p / (opW + 1e-12)
    etaC = 0.6 * p / (opZ + 1e-12)

    obj_hist: List[float] = []
    converged = False
    for it in range(max_iter):
        A_old, B_old, C_old = A.copy(), B.copy(), C.copy()

        D = W @ A @ Z.T + W @ B.T + C @ Z.T
        R = mask * (D - Y)
        A = A - etaA * (W.T @ R @ Z) / p

        D = W @ A @ Z.T + W @ B.T + C @ Z.T
        R = mask * (D - Y)
        B = project_B_to_feasible(B - etaB * ((R.T @ W) / p), PZ_perp)

        D = W @ A @ Z.T + W @ B.T + C @ Z.T
        R = mask * (D - Y)
        C = project_C_to_feasible(C - etaC * ((R @ Z) / p), PW_perp)

        D = W @ A @ Z.T + W @ B.T + C @ Z.T
        obj = 0.5 / p * float(np.sum((mask * (D - Y)) ** 2))
        obj_hist.append(obj)

        rel_change = max(
            fro_norm(A - A_old) / (fro_norm(A_old) + 1e-12),
            fro_norm(B - B_old) / (fro_norm(B_old) + 1e-12),
            fro_norm(C - C_old) / (fro_norm(C_old) + 1e-12),
        )
        if rel_change < tol:
            converged = True
            break

    M_hat = W @ A @ Z.T + W @ B.T + C @ Z.T
    return M_hat, {
        "A": A, "B": B, "C": C,
        "obj": np.asarray(obj_hist),
        "n_iter": len(obj_hist),
        "final_obj": obj_hist[-1] if obj_hist else np.nan,
        "converged": converged,
    }


def fit_lowrank_convex(
    Y: np.ndarray,
    mask: np.ndarray,
    lam: float,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Vanilla nuclear-norm matrix completion baseline."""
    p = max(float(np.mean(mask)), 1e-8)
    L = np.zeros_like(Y)
    eta = 0.8 * p
    obj_hist: List[float] = []
    converged = False
    for _ in range(max_iter):
        L_old = L.copy()
        R = mask * (L - Y)
        G = L - eta * R / p
        L = svt(G, eta * lam / p)
        obj = 0.5 / p * float(np.sum((mask * (L - Y)) ** 2)) + lam / p * float(np.sum(np.linalg.svd(L, compute_uv=False)))
        obj_hist.append(obj)
        if fro_norm(L - L_old) / (fro_norm(L_old) + 1e-12) < tol:
            converged = True
            break
    return L, {
        "L": L,
        "obj": np.asarray(obj_hist),
        "rank": numerical_rank(L),
        "n_iter": len(obj_hist),
        "final_obj": obj_hist[-1] if obj_hist else np.nan,
        "converged": converged,
    }


def spectral_init(Y: np.ndarray, mask: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray]:
    p = max(float(np.mean(mask)), 1e-8)
    M0 = Y / p
    U, s, Vh = np.linalg.svd(M0, full_matrices=False)
    r = min(r, len(s))
    s = np.maximum(s[:r], 0.0)
    X0 = U[:, :r] @ np.diag(np.sqrt(s))
    Y0 = Vh[:r, :].T @ np.diag(np.sqrt(s))
    return X0, Y0


def fit_lowrank_nonconvex(
    Y: np.ndarray,
    mask: np.ndarray,
    r: int,
    lam: float,
    max_iter: int = 400,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Factorized low-rank baseline."""
    p = max(float(np.mean(mask)), 1e-8)
    X, Yf = spectral_init(Y, mask, r)
    obj_hist: List[float] = []
    converged = False
    for _ in range(max_iter):
        X_old, Y_old = X.copy(), Yf.copy()
        D = X @ Yf.T
        R = mask * (D - Y)
        Lx = spectral_norm(Yf.T @ Yf) / p + lam / p
        Ly = spectral_norm(X.T @ X) / p + lam / p
        etaX = 0.7 / (Lx + 1e-12)
        etaY = 0.7 / (Ly + 1e-12)
        X = X - etaX * (R @ Yf / p + lam * X / p)
        Yf = Yf - etaY * (R.T @ X_old / p + lam * Yf / p)
        D = X @ Yf.T
        obj = 0.5 / p * float(np.sum((mask * (D - Y)) ** 2)) + 0.5 * lam / p * (fro_norm(X) ** 2 + fro_norm(Yf) ** 2)
        obj_hist.append(obj)
        rel_change = max(
            fro_norm(X - X_old) / (fro_norm(X_old) + 1e-12),
            fro_norm(Yf - Y_old) / (fro_norm(Y_old) + 1e-12),
        )
        if rel_change < tol:
            converged = True
            break
    M_hat = X @ Yf.T
    return M_hat, {
        "X": X, "Y": Yf,
        "obj": np.asarray(obj_hist),
        "rank": numerical_rank(M_hat),
        "n_iter": len(obj_hist),
        "final_obj": obj_hist[-1] if obj_hist else np.nan,
        "converged": converged,
    }


def fit_additive_no_interaction(
    Y: np.ndarray,
    mask: np.ndarray,
    W: np.ndarray,
    Z: np.ndarray,
    lam: float,
    max_iter: int = 350,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Covariate-adjusted additive model without W A Z^T interaction:
        M = W B^T + C Z^T + L.
    """
    p = max(float(np.mean(mask)), 1e-8)
    PW_perp = projector_onto_orthogonal_complement(W)
    PZ_perp = projector_onto_orthogonal_complement(Z)
    n1, n2 = Y.shape
    d1 = W.shape[1]
    d2 = Z.shape[1]
    B = np.zeros((n2, d1))
    C = np.zeros((n1, d2))
    L = np.zeros_like(Y)

    opW = spectral_norm(W.T @ W)
    opZ = spectral_norm(Z.T @ Z)
    etaB = 0.6 * p / (opW + 1e-12)
    etaC = 0.6 * p / (opZ + 1e-12)
    etaL = 0.5 * p

    obj_hist: List[float] = []
    converged = False
    for _ in range(max_iter):
        B_old, C_old, L_old = B.copy(), C.copy(), L.copy()

        D = W @ B.T + C @ Z.T + L
        R = mask * (D - Y)
        B = project_B_to_feasible(B - etaB * ((R.T @ W) / p), PZ_perp)

        D = W @ B.T + C @ Z.T + L
        R = mask * (D - Y)
        C = project_C_to_feasible(C - etaC * ((R @ Z) / p), PW_perp)

        D = W @ B.T + C @ Z.T + L
        R = mask * (D - Y)
        G = project_L_to_feasible(L - etaL * R / p, PW_perp, PZ_perp)
        L = project_L_to_feasible(svt(G, etaL * lam / p), PW_perp, PZ_perp)

        D = W @ B.T + C @ Z.T + L
        obj = 0.5 / p * float(np.sum((mask * (D - Y)) ** 2)) + lam / p * float(np.sum(np.linalg.svd(L, compute_uv=False)))
        obj_hist.append(obj)
        rel_change = max(
            fro_norm(B - B_old) / (fro_norm(B_old) + 1e-12),
            fro_norm(C - C_old) / (fro_norm(C_old) + 1e-12),
            fro_norm(L - L_old) / (fro_norm(L_old) + 1e-12),
        )
        if rel_change < tol:
            converged = True
            break

    M_hat = W @ B.T + C @ Z.T + L
    return M_hat, {
        "B": B, "C": C, "L": L,
        "obj": np.asarray(obj_hist),
        "rank": numerical_rank(L),
        "n_iter": len(obj_hist),
        "final_obj": obj_hist[-1] if obj_hist else np.nan,
        "converged": converged,
    }


def fit_ours_full(
    Y: np.ndarray,
    mask: np.ndarray,
    W: np.ndarray,
    Z: np.ndarray,
    lam: float,
    max_iter: int = 400,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    The paper's one-stage nuclear-norm estimator:
        M = W A Z^T + W B^T + C Z^T + L,
        penalty = (lambda/p) ||L||_*.
    """
    p = max(float(np.mean(mask)), 1e-8)
    PW_perp = projector_onto_orthogonal_complement(W)
    PZ_perp = projector_onto_orthogonal_complement(Z)
    n1, n2 = Y.shape
    d1 = W.shape[1]
    d2 = Z.shape[1]

    A = np.zeros((d1, d2))
    B = np.zeros((n2, d1))
    C = np.zeros((n1, d2))
    L = np.zeros_like(Y)

    opW = spectral_norm(W.T @ W)
    opZ = spectral_norm(Z.T @ Z)
    etaA = 0.6 * p / (opW * opZ + 1e-12)
    etaB = 0.6 * p / (opW + 1e-12)
    etaC = 0.6 * p / (opZ + 1e-12)
    etaL = 0.5 * p

    obj_hist: List[float] = []
    converged = False
    for _ in range(max_iter):
        A_old, B_old, C_old, L_old = A.copy(), B.copy(), C.copy(), L.copy()

        D = W @ A @ Z.T + W @ B.T + C @ Z.T + L
        R = mask * (D - Y)
        A = A - etaA * (W.T @ R @ Z) / p

        D = W @ A @ Z.T + W @ B.T + C @ Z.T + L
        R = mask * (D - Y)
        B = project_B_to_feasible(B - etaB * ((R.T @ W) / p), PZ_perp)

        D = W @ A @ Z.T + W @ B.T + C @ Z.T + L
        R = mask * (D - Y)
        C = project_C_to_feasible(C - etaC * ((R @ Z) / p), PW_perp)

        D = W @ A @ Z.T + W @ B.T + C @ Z.T + L
        R = mask * (D - Y)
        G = project_L_to_feasible(L - etaL * R / p, PW_perp, PZ_perp)
        L = project_L_to_feasible(svt(G, etaL * lam / p), PW_perp, PZ_perp)

        D = W @ A @ Z.T + W @ B.T + C @ Z.T + L
        obj = 0.5 / p * float(np.sum((mask * (D - Y)) ** 2)) + lam / p * float(np.sum(np.linalg.svd(L, compute_uv=False)))
        obj_hist.append(obj)
        rel_change = max(
            fro_norm(A - A_old) / (fro_norm(A_old) + 1e-12),
            fro_norm(B - B_old) / (fro_norm(B_old) + 1e-12),
            fro_norm(C - C_old) / (fro_norm(C_old) + 1e-12),
            fro_norm(L - L_old) / (fro_norm(L_old) + 1e-12),
        )
        if rel_change < tol:
            converged = True
            break

    M_hat = W @ A @ Z.T + W @ B.T + C @ Z.T + L
    return M_hat, {
        "A": A, "B": B, "C": C, "L": L,
        "obj": np.asarray(obj_hist),
        "rank": numerical_rank(L),
        "n_iter": len(obj_hist),
        "final_obj": obj_hist[-1] if obj_hist else np.nan,
        "converged": converged,
        "lambda": lam,
    }


# ============================================================
# Method comparison helpers
# ============================================================

@dataclass
class MethodSpec:
    name: str
    family: str
    fit_fn: Callable[..., Tuple[np.ndarray, Dict[str, Any]]]
    uses_lambda: bool = True
    lambda_grid: Optional[List[float]] = None


def default_methods() -> List[MethodSpec]:
    return [
        MethodSpec(
            "baseline_cov_interaction_only",
            "cov_interaction_only",
            fit_cov_interaction_only,
            False,
            None,
        ),
        MethodSpec(
            "common_lowrank_convex",
            "lowrank_convex",
            fit_lowrank_convex,
            True,
            [0.25, 0.5, 1.0, 2.0, 4.0],
        ),
        MethodSpec(
            "paper1_lowrank_nonconvex",
            "lowrank_nonconvex",
            fit_lowrank_nonconvex,
            True,
            [0.25, 0.5, 1.0, 2.0, 4.0],
        ),
        MethodSpec(
            "paper2_additive_no_interaction",
            "additive",
            fit_additive_no_interaction,
            True,
            [0.25, 0.5, 1.0, 2.0, 4.0],
        ),
        MethodSpec(
            "ours_full_interaction_latent",
            "full",
            fit_ours_full,
            True,
            [0.25, 0.5, 1.0, 2.0, 4.0],
        ),
    ]


def evaluate_method(
    method: MethodSpec,
    Y_train: np.ndarray,
    train_mask: np.ndarray,
    Y_val_target: np.ndarray,
    val_mask: np.ndarray,
    W: np.ndarray,
    Z: np.ndarray,
    r_true: int,
    lambda_base: float,
    max_iter: int,
    tol: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Method comparison evaluator.

    Important: validation uses Y_val_target, which should be the noisy held-out observations
    in a fair comparison. Do NOT pass the clean M_true unless you intentionally want an oracle
    tuning rule.
    """
    best_score = float("inf")
    best_M: Optional[np.ndarray] = None
    best_info: Dict[str, Any] = {}
    grid = method.lambda_grid if method.lambda_grid is not None else [1.0]

    for mult in grid:
        lam = lambda_base * mult
        if method.family == "cov_interaction_only":
            M_hat, info = method.fit_fn(Y_train, train_mask, W, Z, max_iter=max_iter, tol=tol)
        elif method.family == "lowrank_convex":
            M_hat, info = method.fit_fn(Y_train, train_mask, lam, max_iter=max_iter, tol=tol)
        elif method.family == "lowrank_nonconvex":
            M_hat, info = method.fit_fn(Y_train, train_mask, r_true, lam, max_iter=max_iter, tol=tol)
        elif method.family == "additive":
            M_hat, info = method.fit_fn(Y_train, train_mask, W, Z, lam, max_iter=max_iter, tol=tol)
        elif method.family == "full":
            M_hat, info = method.fit_fn(Y_train, train_mask, W, Z, lam, max_iter=max_iter, tol=tol)
        else:
            raise ValueError(f"Unknown family: {method.family}")

        # Fair tuning: validation against noisy held-out observations, not clean truth.
        val_rmse = rmse_on_mask(M_hat, Y_val_target, val_mask)
        if val_rmse < best_score:
            best_score = val_rmse
            best_M = M_hat
            best_info = {
                "lambda": lam,
                "lambda_mult": mult,
                "val_rmse_noisy": val_rmse,
                **info,
            }

    if best_M is None:
        raise RuntimeError(f"No fit produced for method {method.name}.")
    return best_M, best_info


def summarize_comparison_fit(
    M_hat: np.ndarray,
    M_true: np.ndarray,
    Y_obs: np.ndarray,
    test_mask: np.ndarray,
    method_name: str,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "method": method_name,
        "clean_rel_fro_M": relative_fro_error(M_hat, M_true),
        "clean_inf_M": inf_error(M_hat, M_true),
        "clean_test_rmse": rmse_on_mask(M_hat, M_true, test_mask),
        "noisy_test_rmse": rmse_on_mask(M_hat, Y_obs, test_mask),
    }
    # Copy scalar diagnostics only.
    for k, v in extra.items():
        if isinstance(v, (int, float, np.floating, np.integer, bool)):
            out[k] = float(v)
    return out


# ============================================================
# Theorem-study helpers
# ============================================================

def summarize_ours_components(
    info: Dict[str, Any],
    A_true: np.ndarray,
    B_true: np.ndarray,
    C_true: np.ndarray,
    L_true: np.ndarray,
    M_hat: np.ndarray,
    M_true: np.ndarray,
) -> Dict[str, float]:
    A_hat = info["A"]
    B_hat = info["B"]
    C_hat = info["C"]
    L_hat = info["L"]
    return {
        "A_fro": fro_norm(A_hat - A_true),
        "B_fro": fro_norm(B_hat - B_true),
        "C_fro": fro_norm(C_hat - C_true),
        "L_fro": fro_norm(L_hat - L_true),
        "M_fro": fro_norm(M_hat - M_true),
        "A_rel_fro": relative_fro_error(A_hat, A_true),
        "B_rel_fro": relative_fro_error(B_hat, B_true),
        "C_rel_fro": relative_fro_error(C_hat, C_true),
        "L_rel_fro": relative_fro_error(L_hat, L_true),
        "M_rel_fro": relative_fro_error(M_hat, M_true),
        "B_2inf": norm_2inf(B_hat - B_true),
        "C_2inf": norm_2inf(C_hat - C_true),
        "L_2inf": norm_2inf(L_hat - L_true),
        "M_inf": inf_error(M_hat, M_true),
        "rank_L_hat": float(numerical_rank(L_hat)),
        "n_iter": float(info.get("n_iter", np.nan)),
        "final_obj": float(info.get("final_obj", np.nan)),
        "converged": float(info.get("converged", False)),
    }


def run_one_theorem_setting(
    *,
    n1: int,
    n2: int,
    d1: int,
    d2: int,
    r_true: int,
    p_obs: float,
    sigma: float,
    scale_A: float,
    scale_B: float,
    scale_C: float,
    scale_L: float,
    seed: int,
    c_lambda: float,
    max_iter: int,
    tol: float,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    W, Z, A_true, B_true, C_true, L_true, M_true = generate_truth(
        n1=n1, n2=n2, d1=d1, d2=d2, r_true=r_true, rng=rng,
        scale_A=scale_A, scale_B=scale_B, scale_C=scale_C, scale_L=scale_L,
    )
    Y_obs, mask_obs, _ = generate_noisy_obs(M_true, p_obs, sigma, rng)
    p_fit = max(float(np.mean(mask_obs)), 1e-8)
    lam = recommended_lambda(sigma=sigma, n1=n1, n2=n2, p_fit=p_fit, c_lambda=c_lambda)
    M_hat, info = fit_ours_full(Y_obs, mask_obs, W, Z, lam, max_iter=max_iter, tol=tol)

    row: Dict[str, Any] = summarize_ours_components(info, A_true, B_true, C_true, L_true, M_hat, M_true)
    s = np.linalg.svd(L_true, compute_uv=False)
    row.update({
        "n1": n1, "n2": n2, "d1": d1, "d2": d2, "r_true": r_true,
        "p_obs": p_obs,
        "p_fit": p_fit,
        "sigma": sigma,
        "scale_A": scale_A,
        "scale_B": scale_B,
        "scale_C": scale_C,
        "scale_L": scale_L,
        "seed": seed,
        "lambda": lam,
        "c_lambda": c_lambda,
        "sigma_min_L": float(np.min(s)) if len(s) > 0 else np.nan,
        "sigma_max_L": float(np.max(s)) if len(s) > 0 else np.nan,
        "observed_entries": float(np.sum(mask_obs)),
    })
    return row


# ============================================================
# Experiment runners and plotting
# ============================================================

def run_one_comparison_setting(
    *,
    n1: int,
    n2: int,
    d1: int,
    d2: int,
    r_true: int,
    p_obs: float,
    sigma: float,
    scale_A: float,
    scale_B: float,
    scale_C: float,
    scale_L: float,
    seed: int,
    methods: List[MethodSpec],
    train_frac: float,
    val_frac: float,
    c_lambda: float,
    max_iter: int,
    tol: float,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    W, Z, A_true, B_true, C_true, L_true, M_true = generate_truth(
        n1=n1, n2=n2, d1=d1, d2=d2, r_true=r_true, rng=rng,
        scale_A=scale_A, scale_B=scale_B, scale_C=scale_C, scale_L=scale_L,
    )
    Y_obs, mask_obs, _ = generate_noisy_obs(M_true, p_obs, sigma, rng)
    train_mask, val_mask, test_mask = split_mask(mask_obs, train_frac=train_frac, val_frac=val_frac, rng=rng)
    Y_train = train_mask * Y_obs

    p_train = max(float(np.mean(train_mask)), 1e-8)
    lambda_base = recommended_lambda(sigma=sigma, n1=n1, n2=n2, p_fit=p_train, c_lambda=c_lambda)

    rows: List[Dict[str, Any]] = []
    for method in methods:
        M_hat, extra = evaluate_method(
            method=method,
            Y_train=Y_train,
            train_mask=train_mask,
            Y_val_target=Y_obs,  # fair: noisy held-out observations, not clean M_true
            val_mask=val_mask,
            W=W,
            Z=Z,
            r_true=r_true,
            lambda_base=lambda_base,
            max_iter=max_iter,
            tol=tol,
        )
        row = summarize_comparison_fit(M_hat, M_true, Y_obs, test_mask, method.name, extra)
        row.update({
            "n1": n1, "n2": n2, "d1": d1, "d2": d2, "r_true": r_true,
            "p_obs": p_obs,
            "p_train": p_train,
            "sigma": sigma,
            "scale_A": scale_A,
            "scale_B": scale_B,
            "scale_C": scale_C,
            "scale_L": scale_L,
            "seed": seed,
            "observed_entries": float(np.sum(mask_obs)),
            "train_entries": float(np.sum(train_mask)),
            "val_entries": float(np.sum(val_mask)),
            "test_entries": float(np.sum(test_mask)),
            "lambda_base": lambda_base,
        })
        rows.append(row)
    return rows


def aggregate_and_plot_comparison(df_raw: pd.DataFrame, x_col: str, out_dir: str, prefix: str) -> None:
    summary = (
        df_raw.groupby([x_col, "method"], as_index=False)
        .agg(
            clean_rel_fro_M_mean=("clean_rel_fro_M", "mean"),
            clean_rel_fro_M_std=("clean_rel_fro_M", "std"),
            clean_inf_M_mean=("clean_inf_M", "mean"),
            clean_inf_M_std=("clean_inf_M", "std"),
            clean_test_rmse_mean=("clean_test_rmse", "mean"),
            clean_test_rmse_std=("clean_test_rmse", "std"),
            noisy_test_rmse_mean=("noisy_test_rmse", "mean"),
            noisy_test_rmse_std=("noisy_test_rmse", "std"),
        )
    )
    summary.to_csv(os.path.join(out_dir, f"{prefix}_summary.csv"), index=False)
    df_raw.to_csv(os.path.join(out_dir, f"{prefix}_raw.csv"), index=False)

    method_order = list(summary["method"].unique())
    plots = [
        ("clean_rel_fro_M", "Relative Frobenius error of M (clean truth)", "fro", True),
        ("clean_inf_M", "Infinity error of M (clean truth)", "inf", True),
        ("noisy_test_rmse", "Held-out RMSE (noisy observations)", "rmse_noisy", False),
        ("clean_test_rmse", "Held-out RMSE (clean signal, simulation only)", "rmse_clean", False),
    ]
    for metric, title, fname, logy in plots:
        plt.figure(figsize=(8, 5))
        for method in method_order:
            sub = summary[summary["method"] == method].sort_values(x_col)
            if sub.empty:
                continue
            x = sub[x_col].to_numpy()
            mean = sub[f"{metric}_mean"].to_numpy()
            std = np.nan_to_num(sub[f"{metric}_std"].to_numpy())
            plt.errorbar(x, mean, yerr=std, marker="o", capsize=3, label=method)
        plt.title(title + f" vs {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(metric)
        if logy:
            plt.yscale("log")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_{fname}.png"), dpi=180)
        plt.close()


def aggregate_and_plot_theorem(df_raw: pd.DataFrame, x_col: str, out_dir: str, prefix: str) -> None:
    summary = (
        df_raw.groupby([x_col], as_index=False)
        .agg(
            A_rel_fro_mean=("A_rel_fro", "mean"), A_rel_fro_std=("A_rel_fro", "std"),
            B_rel_fro_mean=("B_rel_fro", "mean"), B_rel_fro_std=("B_rel_fro", "std"),
            C_rel_fro_mean=("C_rel_fro", "mean"), C_rel_fro_std=("C_rel_fro", "std"),
            L_rel_fro_mean=("L_rel_fro", "mean"), L_rel_fro_std=("L_rel_fro", "std"),
            M_rel_fro_mean=("M_rel_fro", "mean"), M_rel_fro_std=("M_rel_fro", "std"),
            B_2inf_mean=("B_2inf", "mean"), B_2inf_std=("B_2inf", "std"),
            C_2inf_mean=("C_2inf", "mean"), C_2inf_std=("C_2inf", "std"),
            L_2inf_mean=("L_2inf", "mean"), L_2inf_std=("L_2inf", "std"),
            M_inf_mean=("M_inf", "mean"), M_inf_std=("M_inf", "std"),
            rank_L_hat_mean=("rank_L_hat", "mean"), rank_L_hat_std=("rank_L_hat", "std"),
        )
    )
    summary.to_csv(os.path.join(out_dir, f"{prefix}_summary.csv"), index=False)
    df_raw.to_csv(os.path.join(out_dir, f"{prefix}_raw.csv"), index=False)

    plot_specs = [
        (["A_rel_fro", "B_rel_fro", "C_rel_fro", "L_rel_fro", "M_rel_fro"], "Relative Frobenius errors", "rel_fro", True),
        (["B_2inf", "C_2inf", "L_2inf", "M_inf"], "Infinity-type errors", "inf_type", True),
        (["rank_L_hat"], "Estimated rank of L", "rank", False),
    ]
    for metrics, title, fname, logy in plot_specs:
        plt.figure(figsize=(8, 5))
        for metric in metrics:
            x = summary[x_col].to_numpy()
            mean = summary[f"{metric}_mean"].to_numpy()
            std = np.nan_to_num(summary[f"{metric}_std"].to_numpy())
            plt.errorbar(x, mean, yerr=std, marker="o", capsize=3, label=metric)
        plt.title(title + f" vs {x_col}")
        plt.xlabel(x_col)
        plt.ylabel("error")
        if logy:
            plt.yscale("log")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_{fname}.png"), dpi=180)
        plt.close()


def run_comparison_experiment(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)
    methods = default_methods()

    if args.quick:
        repeats = 1
        n1, n2 = 35, 40
        d1, d2, r_true = 2, 2, 1
        max_iter = min(args.max_iter, 150)
    else:
        repeats = args.repeats
        n1, n2 = args.n1, args.n2
        d1, d2, r_true = args.d1, args.d2, args.r_true
        max_iter = args.max_iter

    rows: List[Dict[str, Any]] = []

    if args.mode == "vary_p":
        grid = args.p_grid
        for p_obs in grid:
            for rep in range(repeats):
                rows.extend(run_one_comparison_setting(
                    n1=n1, n2=n2, d1=d1, d2=d2, r_true=r_true,
                    p_obs=p_obs, sigma=args.sigma,
                    scale_A=args.scale_A, scale_B=args.scale_B, scale_C=args.scale_C, scale_L=args.scale_L,
                    seed=args.seed + 1000 * rep + int(1000 * p_obs),
                    methods=methods,
                    train_frac=args.train_frac, val_frac=args.val_frac,
                    c_lambda=args.c_lambda,
                    max_iter=max_iter,
                    tol=args.tol,
                ))
                print(f"[comparison] done p_obs={p_obs}, rep={rep+1}/{repeats}")
        df = pd.DataFrame(rows)
        aggregate_and_plot_comparison(df, x_col="p_obs", out_dir=args.out_dir, prefix="compare_vary_p")

    elif args.mode == "vary_interaction":
        grid = args.scaleA_grid
        for scale_A in grid:
            for rep in range(repeats):
                rows.extend(run_one_comparison_setting(
                    n1=n1, n2=n2, d1=d1, d2=d2, r_true=r_true,
                    p_obs=args.p_obs, sigma=args.sigma,
                    scale_A=scale_A, scale_B=args.scale_B, scale_C=args.scale_C, scale_L=args.scale_L,
                    seed=args.seed + 1000 * rep + int(1000 * scale_A),
                    methods=methods,
                    train_frac=args.train_frac, val_frac=args.val_frac,
                    c_lambda=args.c_lambda,
                    max_iter=max_iter,
                    tol=args.tol,
                ))
                print(f"[comparison] done scale_A={scale_A}, rep={rep+1}/{repeats}")
        df = pd.DataFrame(rows)
        aggregate_and_plot_comparison(df, x_col="scale_A", out_dir=args.out_dir, prefix="compare_vary_interaction")

    elif args.mode == "vary_latent":
        grid = args.scaleL_grid
        for scale_L in grid:
            for rep in range(repeats):
                rows.extend(run_one_comparison_setting(
                    n1=n1, n2=n2, d1=d1, d2=d2, r_true=r_true,
                    p_obs=args.p_obs, sigma=args.sigma,
                    scale_A=args.scale_A, scale_B=args.scale_B, scale_C=args.scale_C, scale_L=scale_L,
                    seed=args.seed + 1000 * rep + int(1000 * scale_L),
                    methods=methods,
                    train_frac=args.train_frac, val_frac=args.val_frac,
                    c_lambda=args.c_lambda,
                    max_iter=max_iter,
                    tol=args.tol,
                ))
                print(f"[comparison] done scale_L={scale_L}, rep={rep+1}/{repeats}")
        df = pd.DataFrame(rows)
        aggregate_and_plot_comparison(df, x_col="scale_L", out_dir=args.out_dir, prefix="compare_vary_latent")

    else:
        raise ValueError("For --study comparison, mode must be vary_p, vary_interaction, or vary_latent.")


def run_theorem_experiment(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)
    if args.quick:
        repeats = 1
        n1, n2 = 35, 40
        d1, d2, r_true = 2, 2, 1
        max_iter = min(args.max_iter, 150)
    else:
        repeats = args.repeats
        n1, n2 = args.n1, args.n2
        d1, d2, r_true = args.d1, args.d2, args.r_true
        max_iter = args.max_iter

    rows: List[Dict[str, Any]] = []

    if args.mode == "vary_p":
        grid = args.p_grid
        for p_obs in grid:
            for rep in range(repeats):
                rows.append(run_one_theorem_setting(
                    n1=n1, n2=n2, d1=d1, d2=d2, r_true=r_true,
                    p_obs=p_obs, sigma=args.sigma,
                    scale_A=args.scale_A, scale_B=args.scale_B, scale_C=args.scale_C, scale_L=args.scale_L,
                    seed=args.seed + 1000 * rep + int(1000 * p_obs),
                    c_lambda=args.c_lambda,
                    max_iter=max_iter,
                    tol=args.tol,
                ))
                print(f"[theorem] done p_obs={p_obs}, rep={rep+1}/{repeats}")
        df = pd.DataFrame(rows)
        aggregate_and_plot_theorem(df, x_col="p_obs", out_dir=args.out_dir, prefix="theorem_vary_p")

    elif args.mode == "vary_sigma":
        grid = args.sigma_grid
        for sigma in grid:
            for rep in range(repeats):
                rows.append(run_one_theorem_setting(
                    n1=n1, n2=n2, d1=d1, d2=d2, r_true=r_true,
                    p_obs=args.p_obs, sigma=sigma,
                    scale_A=args.scale_A, scale_B=args.scale_B, scale_C=args.scale_C, scale_L=args.scale_L,
                    seed=args.seed + 1000 * rep + int(100000 * sigma),
                    c_lambda=args.c_lambda,
                    max_iter=max_iter,
                    tol=args.tol,
                ))
                print(f"[theorem] done sigma={sigma}, rep={rep+1}/{repeats}")
        df = pd.DataFrame(rows)
        aggregate_and_plot_theorem(df, x_col="sigma", out_dir=args.out_dir, prefix="theorem_vary_sigma")

    elif args.mode == "vary_n":
        grid = args.n_grid
        for n in grid:
            for rep in range(repeats):
                rows.append(run_one_theorem_setting(
                    n1=n, n2=n, d1=d1, d2=d2, r_true=r_true,
                    p_obs=args.p_obs, sigma=args.sigma,
                    scale_A=args.scale_A, scale_B=args.scale_B, scale_C=args.scale_C, scale_L=args.scale_L,
                    seed=args.seed + 1000 * rep + int(n),
                    c_lambda=args.c_lambda,
                    max_iter=max_iter,
                    tol=args.tol,
                ))
                print(f"[theorem] done n={n}, rep={rep+1}/{repeats}")
        df = pd.DataFrame(rows)
        aggregate_and_plot_theorem(df, x_col="n1", out_dir=args.out_dir, prefix="theorem_vary_n")

    else:
        raise ValueError("For --study theorem, mode must be vary_p, vary_sigma, or vary_n.")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refined compare1 experiments for generalized matrix completion")
    parser.add_argument("--study", type=str, default="comparison", choices=["comparison", "theorem"],
                        help="comparison = tuned method comparison; theorem = fixed-lambda theorem check")
    parser.add_argument("--mode", type=str, default="vary_p",
                        help="comparison: vary_p, vary_interaction, vary_latent; theorem: vary_p, vary_sigma, vary_n")
    parser.add_argument("--out_dir", type=str, default="compare1_refined_outputs")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--n1", type=int, default=80)
    parser.add_argument("--n2", type=int, default=90)
    parser.add_argument("--d1", type=int, default=4)
    parser.add_argument("--d2", type=int, default=5)
    parser.add_argument("--r_true", type=int, default=3)

    parser.add_argument("--p_obs", type=float, default=0.35)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--scale_A", type=float, default=1.0)
    parser.add_argument("--scale_B", type=float, default=1.0)
    parser.add_argument("--scale_C", type=float, default=1.0)
    parser.add_argument("--scale_L", type=float, default=1.0)
    parser.add_argument("--c_lambda", type=float, default=5.0)

    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--val_frac", type=float, default=0.15)

    parser.add_argument("--max_iter", type=int, default=400)
    parser.add_argument("--tol", type=float, default=1e-6)

    parser.add_argument("--p_grid", type=float, nargs="*", default=[0.15, 0.25, 0.35, 0.45, 0.55])
    parser.add_argument("--scaleA_grid", type=float, nargs="*", default=[0.0, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--scaleL_grid", type=float, nargs="*", default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--sigma_grid", type=float, nargs="*", default=[0.01, 0.03, 0.05, 0.08, 0.10])
    parser.add_argument("--n_grid", type=int, nargs="*", default=[50, 60, 70, 80, 90])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.study == "comparison":
        run_comparison_experiment(args)
    elif args.study == "theorem":
        run_theorem_experiment(args)
    else:
        raise ValueError(f"Unknown study: {args.study}")


if __name__ == "__main__":
    main()
