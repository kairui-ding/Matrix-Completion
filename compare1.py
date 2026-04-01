import argparse
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Method comparison experiments for generalized matrix completion
#
# Model:
#   M* = W A Z^T + W B^T + C Z^T + L
#
# Compared methods:
#   1. cov_only            : covariates only, no latent part
#   2. lowrank_convex      : vanilla nuclear-norm matrix completion
#   3. lowrank_nonconvex   : factorized low-rank matrix completion (Chen et al.-style)
#   4. additive_no_inter   : covariate-adjusted model without interaction term WA Z^T
#   5. ours_full           : full interaction + additive + latent model
#
# Main recommended plots:
#   - vary_p: compare methods as observation rate changes
#   - vary_interaction: compare methods as ||A*|| changes
#
# This script is designed to be reasonably fast for moderate n.
# ============================================================


# ---------------------------
# Utility functions
# ---------------------------

def fro_norm(X: np.ndarray) -> float:
    return float(np.linalg.norm(X, ord="fro"))


def spectral_norm(X: np.ndarray) -> float:
    return float(np.linalg.norm(X, ord=2))


def relative_fro_error(X_hat: np.ndarray, X_true: np.ndarray, eps: float = 1e-12) -> float:
    return fro_norm(X_hat - X_true) / (fro_norm(X_true) + eps)


def inf_error(X_hat: np.ndarray, X_true: np.ndarray) -> float:
    return float(np.max(np.abs(X_hat - X_true)))


def rmse_on_mask(X_hat: np.ndarray, X_true: np.ndarray, mask: np.ndarray) -> float:
    denom = float(np.sum(mask))
    if denom <= 0:
        return float("nan")
    err = mask * (X_hat - X_true)
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


def rank_r_projection(Y: np.ndarray, r: int) -> np.ndarray:
    U, s, Vh = np.linalg.svd(Y, full_matrices=False)
    r = min(r, len(s))
    return U[:, :r] @ np.diag(s[:r]) @ Vh[:r, :]


def numerical_rank(X: np.ndarray, rel_thresh: float = 1e-3) -> int:
    s = np.linalg.svd(X, compute_uv=False)
    if s.size == 0 or s[0] <= 1e-15:
        return 0
    return int(np.sum(s > rel_thresh * s[0]))


def recommended_lambda(sigma: float, n1: int, n2: int, p_obs: float, c_lambda: float = 5.0) -> float:
    return c_lambda * sigma * math.sqrt(max(n1, n2) * p_obs)


# ---------------------------
# Data generation
# ---------------------------

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    W = rng.standard_normal((n1, d1)) / math.sqrt(n1)
    Z = rng.standard_normal((n2, d2)) / math.sqrt(n2)

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


def generate_noisy_obs(M_true: np.ndarray, p_obs: float, sigma: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    noise = rng.normal(0.0, sigma, size=M_true.shape)
    observed = (rng.random(M_true.shape) < p_obs).astype(float)
    Y_obs = observed * (M_true + noise)
    return Y_obs, observed


def split_mask(mask_obs: np.ndarray, train_frac: float, val_frac: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


# ---------------------------
# Solvers
# ---------------------------

def fit_cov_only(
    Y: np.ndarray,
    mask: np.ndarray,
    W: np.ndarray,
    Z: np.ndarray,
    max_iter: int = 250,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
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

    obj_hist = []
    for _ in range(max_iter):
        A_old, B_old, C_old = A.copy(), B.copy(), C.copy()

        D = W @ A @ Z.T + W @ B.T + C @ Z.T
        R = mask * (D - Y)
        A = A - etaA * (W.T @ R @ Z) / p

        D = W @ A @ Z.T + W @ B.T + C @ Z.T
        R = mask * (D - Y)
        grad_B = (R.T @ W) / p
        B = project_B_to_feasible(B - etaB * grad_B, PZ_perp)

        D = W @ A @ Z.T + W @ B.T + C @ Z.T
        R = mask * (D - Y)
        grad_C = (R @ Z) / p
        C = project_C_to_feasible(C - etaC * grad_C, PW_perp)

        D = W @ A @ Z.T + W @ B.T + C @ Z.T
        obj = 0.5 / p * float(np.sum((mask * (D - Y)) ** 2))
        obj_hist.append(obj)

        rel_change = max(
            fro_norm(A - A_old) / (fro_norm(A_old) + 1e-12),
            fro_norm(B - B_old) / (fro_norm(B_old) + 1e-12),
            fro_norm(C - C_old) / (fro_norm(C_old) + 1e-12),
        )
        if rel_change < tol:
            break

    M_hat = W @ A @ Z.T + W @ B.T + C @ Z.T
    return M_hat, {"A": A, "B": B, "C": C, "obj": np.asarray(obj_hist)}



def fit_lowrank_convex(
    Y: np.ndarray,
    mask: np.ndarray,
    lam: float,
    max_iter: int = 250,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    p = max(float(np.mean(mask)), 1e-8)
    L = np.zeros_like(Y)
    eta = 0.8 * p
    obj_hist = []
    for _ in range(max_iter):
        L_old = L.copy()
        R = mask * (L - Y)
        G = L - eta * R / p
        L = svt(G, eta * lam / p)
        obj = 0.5 / p * float(np.sum((mask * (L - Y)) ** 2)) + lam / p * float(np.sum(np.linalg.svd(L, compute_uv=False)))
        obj_hist.append(obj)
        rel_change = fro_norm(L - L_old) / (fro_norm(L_old) + 1e-12)
        if rel_change < tol:
            break
    return L, {"L": L, "obj": np.asarray(obj_hist), "rank": numerical_rank(L)}



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
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    p = max(float(np.mean(mask)), 1e-8)
    X, Zf = spectral_init(Y, mask, r)
    obj_hist = []
    for _ in range(max_iter):
        X_old, Z_old = X.copy(), Zf.copy()
        D = X @ Zf.T
        R = mask * (D - Y)
        Lx = spectral_norm(Zf.T @ Zf) / p + lam / p
        Ly = spectral_norm(X.T @ X) / p + lam / p
        etaX = 0.7 / (Lx + 1e-12)
        etaY = 0.7 / (Ly + 1e-12)
        grad_X = R @ Zf / p + lam * X / p
        grad_Y = R.T @ X / p + lam * Zf / p
        X = X - etaX * grad_X
        Zf = Zf - etaY * grad_Y
        D = X @ Zf.T
        obj = 0.5 / p * float(np.sum((mask * (D - Y)) ** 2)) + 0.5 * lam / p * (fro_norm(X) ** 2 + fro_norm(Zf) ** 2)
        obj_hist.append(obj)
        rel_change = max(
            fro_norm(X - X_old) / (fro_norm(X_old) + 1e-12),
            fro_norm(Zf - Z_old) / (fro_norm(Z_old) + 1e-12),
        )
        if rel_change < tol:
            break
    return X @ Zf.T, {"X": X, "Y": Zf, "obj": np.asarray(obj_hist), "rank": numerical_rank(X @ Zf.T)}



def fit_additive_no_interaction(
    Y: np.ndarray,
    mask: np.ndarray,
    W: np.ndarray,
    Z: np.ndarray,
    lam: float,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
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

    obj_hist = []
    for _ in range(max_iter):
        B_old, C_old, L_old = B.copy(), C.copy(), L.copy()
        D = W @ B.T + C @ Z.T + L
        R = mask * (D - Y)
        grad_B = (R.T @ W) / p
        B = project_B_to_feasible(B - etaB * grad_B, PZ_perp)

        D = W @ B.T + C @ Z.T + L
        R = mask * (D - Y)
        grad_C = (R @ Z) / p
        C = project_C_to_feasible(C - etaC * grad_C, PW_perp)

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
            break

    M_hat = W @ B.T + C @ Z.T + L
    return M_hat, {"B": B, "C": C, "L": L, "obj": np.asarray(obj_hist), "rank": numerical_rank(L)}



def fit_ours_full(
    Y: np.ndarray,
    mask: np.ndarray,
    W: np.ndarray,
    Z: np.ndarray,
    lam: float,
    max_iter: int = 350,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
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

    obj_hist = []
    for _ in range(max_iter):
        A_old, B_old, C_old, L_old = A.copy(), B.copy(), C.copy(), L.copy()

        D = W @ A @ Z.T + W @ B.T + C @ Z.T + L
        R = mask * (D - Y)
        A = A - etaA * (W.T @ R @ Z) / p

        D = W @ A @ Z.T + W @ B.T + C @ Z.T + L
        R = mask * (D - Y)
        grad_B = (R.T @ W) / p
        B = project_B_to_feasible(B - etaB * grad_B, PZ_perp)

        D = W @ A @ Z.T + W @ B.T + C @ Z.T + L
        R = mask * (D - Y)
        grad_C = (R @ Z) / p
        C = project_C_to_feasible(C - etaC * grad_C, PW_perp)

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
            break

    M_hat = W @ A @ Z.T + W @ B.T + C @ Z.T + L
    return M_hat, {"A": A, "B": B, "C": C, "L": L, "obj": np.asarray(obj_hist), "rank": numerical_rank(L)}


# ---------------------------
# Tuning and evaluation
# ---------------------------
@dataclass
class MethodSpec:
    name: str
    family: str
    fit_fn: Callable
    uses_lambda: bool = True
    lambda_grid: Optional[List[float]] = None



def evaluate_method(
    method: MethodSpec,
    Y_train: np.ndarray,
    train_mask: np.ndarray,
    Y_val: np.ndarray,
    val_mask: np.ndarray,
    Y_all: np.ndarray,
    W: np.ndarray,
    Z: np.ndarray,
    sigma: float,
    r_true: int,
    lambda_base: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    best_score = float("inf")
    best_M = None
    best_info: Dict[str, object] = {}
    grid = method.lambda_grid if method.lambda_grid is not None else [1.0]

    for mult in grid:
        lam = lambda_base * mult
        if method.family == "cov_only":
            M_hat, info = method.fit_fn(Y_train, train_mask, W, Z)
        elif method.family == "lowrank_convex":
            M_hat, info = method.fit_fn(Y_train, train_mask, lam)
        elif method.family == "lowrank_nonconvex":
            M_hat, info = method.fit_fn(Y_train, train_mask, r_true, lam)
        elif method.family == "additive":
            M_hat, info = method.fit_fn(Y_train, train_mask, W, Z, lam)
        elif method.family == "full":
            M_hat, info = method.fit_fn(Y_train, train_mask, W, Z, lam)
        else:
            raise ValueError(f"Unknown family: {method.family}")

        val_rmse = rmse_on_mask(M_hat, Y_all, val_mask)
        if val_rmse < best_score:
            best_score = val_rmse
            best_M = M_hat
            best_info = {"lambda": lam, "lambda_mult": mult, "val_rmse": val_rmse, **info}

    assert best_M is not None
    return best_M, best_info



def summarize_fit(
    M_hat: np.ndarray,
    M_true: np.ndarray,
    test_mask: np.ndarray,
    method_name: str,
    extra: Dict[str, object],
) -> Dict[str, object]:
    out: Dict[str, object] = {
        "method": method_name,
        "full_rel_fro": relative_fro_error(M_hat, M_true),
        "full_inf": inf_error(M_hat, M_true),
        "test_rmse": rmse_on_mask(M_hat, M_true, test_mask),
    }
    for k, v in extra.items():
        if isinstance(v, (int, float, np.floating, np.integer)):
            out[k] = float(v)
    return out


# ---------------------------
# Experiment runners
# ---------------------------

def default_methods() -> List[MethodSpec]:
    return [
        # MethodSpec("common_lowrank_convex", "lowrank_convex", fit_lowrank_convex, True, [0.5, 1.0, 2.0, 4.0]),
        MethodSpec("paper1_lowrank_nonconvex", "lowrank_nonconvex", fit_lowrank_nonconvex, True, [0.5, 1.0, 2.0, 4.0]),
        MethodSpec("paper2_additive_no_inter", "additive", fit_additive_no_interaction, True, [0.5, 1.0, 2.0, 4.0]),
        MethodSpec("ours_full_interaction", "full", fit_ours_full, True, [0.5, 1.0, 2.0, 4.0]),
    ]



def run_one_setting(
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
) -> List[Dict[str, object]]:
    rng = np.random.default_rng(seed)
    W, Z, A_true, B_true, C_true, L_true, M_true = generate_truth(
        n1=n1, n2=n2, d1=d1, d2=d2, r_true=r_true, rng=rng,
        scale_A=scale_A, scale_B=scale_B, scale_C=scale_C, scale_L=scale_L,
    )
    Y_obs, mask_obs = generate_noisy_obs(M_true, p_obs, sigma, rng)
    train_mask, val_mask, test_mask = split_mask(mask_obs, train_frac=train_frac, val_frac=val_frac, rng=rng)
    Y_train = train_mask * Y_obs
    Y_val = val_mask * Y_obs

    lambda_base = recommended_lambda(sigma=sigma, n1=n1, n2=n2, p_obs=max(float(np.mean(train_mask)), 1e-8), c_lambda=5.0)

    rows = []
    for method in methods:
        M_hat, extra = evaluate_method(
            method=method,
            Y_train=Y_train,
            train_mask=train_mask,
            Y_val=Y_val,
            val_mask=val_mask,
            Y_all=M_true,
            W=W,
            Z=Z,
            sigma=sigma,
            r_true=r_true,
            lambda_base=lambda_base,
        )
        row = summarize_fit(M_hat, M_true, test_mask, method.name, extra)
        row.update({
            "n1": n1,
            "n2": n2,
            "d1": d1,
            "d2": d2,
            "r_true": r_true,
            "p_obs": p_obs,
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
        })
        rows.append(row)
    return rows



def aggregate_and_plot(df_raw: pd.DataFrame, x_col: str, out_dir: str, prefix: str) -> None:
    summary = (
        df_raw.groupby([x_col, "method"], as_index=False)
        .agg(
            full_rel_fro_mean=("full_rel_fro", "mean"),
            full_rel_fro_std=("full_rel_fro", "std"),
            full_inf_mean=("full_inf", "mean"),
            full_inf_std=("full_inf", "std"),
            test_rmse_mean=("test_rmse", "mean"),
            test_rmse_std=("test_rmse", "std"),
        )
    )
    summary.to_csv(os.path.join(out_dir, f"{prefix}_summary.csv"), index=False)
    df_raw.to_csv(os.path.join(out_dir, f"{prefix}_raw.csv"), index=False)

    method_order = list(summary["method"].unique())

    for metric, title, fname in [
        ("full_rel_fro", "Relative Frobenius error of M", "fro"),
        ("full_inf", "Infinity error of M", "inf"),
        ("test_rmse", "Held-out RMSE on clean signal", "rmse"),
    ]:
        plt.figure(figsize=(8, 5))
        for method in method_order:
            sub = summary[summary["method"] == method].sort_values(x_col)
            mean = sub[f"{metric}_mean"].to_numpy()
            std = np.nan_to_num(sub[f"{metric}_std"].to_numpy())
            x = sub[x_col].to_numpy()
            plt.errorbar(x, mean, yerr=std, marker="o", capsize=3, label=method)
        plt.title(title + f" vs {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(metric)
        if metric != "test_rmse":
            plt.yscale("log")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_{fname}.png"), dpi=180)
        plt.close()



def run_experiment(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    methods = default_methods()

    if args.quick:
        repeats = 2
        n1, n2 = 50, 60
        d1, d2, r_true = 3, 4, 2
    else:
        repeats = args.repeats
        n1, n2 = args.n1, args.n2
        d1, d2, r_true = args.d1, args.d2, args.r_true

    scale_B = args.scale_B
    scale_C = args.scale_C
    scale_L = args.scale_L
    sigma = args.sigma

    rows: List[Dict[str, object]] = []

    if args.mode == "vary_p":
        grid = args.p_grid
        for p_obs in grid:
            for rep in range(repeats):
                rows.extend(run_one_setting(
                    n1=n1, n2=n2, d1=d1, d2=d2, r_true=r_true,
                    p_obs=p_obs, sigma=sigma,
                    scale_A=args.scale_A, scale_B=scale_B, scale_C=scale_C, scale_L=scale_L,
                    seed=args.seed + 1000 * rep + int(100 * p_obs),
                    methods=methods,
                    train_frac=args.train_frac, val_frac=args.val_frac,
                ))
                print(f"done p={p_obs}, rep={rep+1}/{repeats}")
        df = pd.DataFrame(rows)
        aggregate_and_plot(df, x_col="p_obs", out_dir=args.out_dir, prefix="compare_vary_p")

    elif args.mode == "vary_interaction":
        grid = args.scaleA_grid
        for scale_A in grid:
            for rep in range(repeats):
                rows.extend(run_one_setting(
                    n1=n1, n2=n2, d1=d1, d2=d2, r_true=r_true,
                    p_obs=args.p_obs, sigma=sigma,
                    scale_A=scale_A, scale_B=scale_B, scale_C=scale_C, scale_L=scale_L,
                    seed=args.seed + 1000 * rep + int(100 * scale_A),
                    methods=methods,
                    train_frac=args.train_frac, val_frac=args.val_frac,
                ))
                print(f"done scale_A={scale_A}, rep={rep+1}/{repeats}")
        df = pd.DataFrame(rows)
        aggregate_and_plot(df, x_col="scale_A", out_dir=args.out_dir, prefix="compare_vary_interaction")

    else:
        raise ValueError("mode must be one of {'vary_p', 'vary_interaction'}")


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Method comparison experiments for generalized matrix completion")
    parser.add_argument("--mode", type=str, default="vary_p", choices=["vary_p", "vary_interaction"])
    parser.add_argument("--out_dir", type=str, default="comparison_outputs")
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

    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)

    parser.add_argument("--p_grid", type=float, nargs="*", default=[0.15, 0.25, 0.35, 0.45, 0.55])
    parser.add_argument("--scaleA_grid", type=float, nargs="*", default=[0.0, 0.5, 1.0, 1.5, 2.0])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
