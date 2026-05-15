#!/usr/bin/env python3
"""
Refined real-data experiment for the covariate-assisted matrix completion model.

This script is designed to match the paper's one-stage estimator

    M = W A Z^T + W B^T + C Z^T + L,

with the constraints

    Z^T B = 0,  W^T C = 0,  W^T L = 0,  L Z = 0,

by reusing the estimator implementations in compare1_refined.py.

Expected data format: GDSC-style long table with columns
    CELL_LINE_NAME, DRUG_ID, LN_IC50, TCGA_DESC, TARGET_PATHWAY

Usage example:
    python bio_data_experiment_refined.py \
        --data_path GDSC_DATASET.csv \
        --n_rows_keep 800 --n_cols_keep 200 \
        --train_frac 0.50 --val_frac 0.20 \
        --out_dir gdsc_refined_out

For sparse-matrix experiments on an otherwise dense GDSC block, use a smaller
--train_frac, e.g. 0.05, 0.10, 0.20.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Import the refined implementations that enforce the paper's projections.
from compare1_refined import (
    default_methods,
    evaluate_method,
    recommended_lambda,
    rmse_on_mask,
    numerical_rank,
    split_mask,
)


def extract_dense_subset(
    df: pd.DataFrame,
    row_col: str = "CELL_LINE_NAME",
    col_col: str = "DRUG_ID",
    response_col: str = "LN_IC50",
    row_feat_col: str = "TCGA_DESC",
    col_feat_col: str = "TARGET_PATHWAY",
    n_rows_keep: int = 800,
    n_cols_keep: int = 200,
    n_iter: int = 6,
    verbose: bool = True,
) -> pd.DataFrame:
    """Extract a relatively dense rectangular block by alternating top-count filtering."""
    use_cols = [row_col, col_col, response_col, row_feat_col, col_feat_col]
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_sub = df[use_cols].dropna(subset=use_cols).copy()
    df_sub[col_col] = df_sub[col_col].astype(str)

    # Average repeated measurements for the same matrix entry and same metadata.
    df_sub = (
        df_sub.groupby([row_col, col_col, row_feat_col, col_feat_col], as_index=False)
              .agg({response_col: "mean"})
    )

    if verbose:
        print("Initial long-table shape:", df_sub.shape)
        print(f"Initial unique rows={df_sub[row_col].nunique()}, unique cols={df_sub[col_col].nunique()}")

    for t in range(n_iter):
        top_rows = df_sub[row_col].value_counts().index[:n_rows_keep]
        df_sub = df_sub[df_sub[row_col].isin(top_rows)].copy()

        top_cols = df_sub[col_col].value_counts().index[:n_cols_keep]
        df_sub = df_sub[df_sub[col_col].isin(top_cols)].copy()

        if verbose:
            nr = df_sub[row_col].nunique()
            nc = df_sub[col_col].nunique()
            dens = len(df_sub) / max(nr * nc, 1)
            print(f"Iter {t+1}: rows={nr}, cols={nc}, obs={len(df_sub)}, density={dens:.4f}")

    return df_sub


def build_matrix_and_covariates(
    df_dense: pd.DataFrame,
    row_col: str = "CELL_LINE_NAME",
    col_col: str = "DRUG_ID",
    response_col: str = "LN_IC50",
    row_feat_col: str = "TCGA_DESC",
    col_feat_col: str = "TARGET_PATHWAY",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[str], List[str]]:
    """Build Y, observation mask, row covariates W, column covariates Z."""
    Y_df = df_dense.pivot(index=row_col, columns=col_col, values=response_col)
    mask = Y_df.notna().astype(float).to_numpy(dtype=float)
    Y_raw = Y_df.fillna(0.0).to_numpy(dtype=float)

    row_names = list(Y_df.index)
    col_names = list(Y_df.columns)

    row_meta = (
        df_dense[[row_col, row_feat_col]]
        .drop_duplicates(subset=[row_col])
        .set_index(row_col)
        .loc[row_names]
    )
    W_df = pd.get_dummies(row_meta[row_feat_col], prefix=row_feat_col, dtype=float)
    W = W_df.to_numpy(dtype=float)

    col_meta = (
        df_dense[[col_col, col_feat_col]]
        .drop_duplicates(subset=[col_col])
        .set_index(col_col)
        .loc[col_names]
    )
    Z_df = pd.get_dummies(col_meta[col_feat_col], prefix=col_feat_col, dtype=float)
    Z = Z_df.to_numpy(dtype=float)

    return Y_raw, mask, W, Z, row_names, col_names, list(W_df.columns), list(Z_df.columns)


def standardize_from_train(Y_raw: np.ndarray, mask_train: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Standardize all observed values using training entries only.
    This avoids test-set leakage.
    """
    train_vals = Y_raw[mask_train > 0.5]
    if train_vals.size == 0:
        raise ValueError("No training entries available for standardization.")
    mean = float(np.mean(train_vals))
    std = float(np.std(train_vals))
    if std <= 1e-12:
        std = 1.0
    Y_std = np.zeros_like(Y_raw, dtype=float)
    observed = Y_raw != 0  # only used to avoid changing stored missing zeros unnecessarily
    Y_std[observed] = (Y_raw[observed] - mean) / std
    # Values in truly missing positions are irrelevant because masks are zero.
    return Y_std, mean, std


def summarize_real_fit(method: str, M_hat: np.ndarray, Y: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray, info: Dict) -> Dict:
    out = {
        "method": method,
        "train_rmse": rmse_on_mask(M_hat, Y, train_mask),
        "val_rmse": rmse_on_mask(M_hat, Y, val_mask),
        "test_rmse": rmse_on_mask(M_hat, Y, test_mask),
    }
    for k, v in info.items():
        if isinstance(v, (int, float, np.floating, np.integer, bool)):
            out[k] = float(v)
    # Add latent rank if L is available.
    if "L" in info:
        out["rank_L"] = float(numerical_rank(info["L"]))
        out["L_fro"] = float(np.linalg.norm(info["L"], ord="fro"))
    return out


def run_real_data_experiment(args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.data_path)
    df_dense = extract_dense_subset(
        df,
        n_rows_keep=args.n_rows_keep,
        n_cols_keep=args.n_cols_keep,
        n_iter=args.n_iter,
        verbose=True,
    )
    Y_raw, mask_obs, W, Z, row_names, col_names, W_names, Z_names = build_matrix_and_covariates(df_dense)

    print("\nConstructed matrix/covariates")
    print("Y_raw shape:", Y_raw.shape)
    print("Observed fraction in selected block:", float(mask_obs.mean()))
    print("W shape:", W.shape, "Z shape:", Z.shape)

    train_mask, val_mask, test_mask = split_mask(mask_obs, args.train_frac, args.val_frac, rng)
    Y, y_mean, y_std = standardize_from_train(Y_raw, train_mask)
    Y_train = Y * train_mask

    print("\nSplit / standardization")
    print("train density:", float(train_mask.mean()))
    print("val density  :", float(val_mask.mean()))
    print("test density :", float(test_mask.mean()))
    print("train mean/std used for LN_IC50:", y_mean, y_std)

    # On real data sigma is unknown. Use unit scale after standardization.
    # Validation will choose the multiplier, so this is only a base scale.
    p_fit = max(float(train_mask.mean()), 1e-8)
    lambda_base = recommended_lambda(
        sigma=args.sigma_base,
        n1=Y.shape[0],
        n2=Y.shape[1],
        p_fit=p_fit,
        c_lambda=args.c_lambda,
    )
    print("lambda_base:", lambda_base)

    methods = default_methods()
    rows = []
    for method in methods:
        print(f"\nFitting {method.name} ...")
        M_hat, info = evaluate_method(
            method=method,
            Y_train=Y_train,
            train_mask=train_mask,
            Y_val_target=Y,
            val_mask=val_mask,
            W=W,
            Z=Z,
            r_true=args.rank,
            lambda_base=lambda_base,
            max_iter=args.max_iter,
            tol=args.tol,
        )
        row = summarize_real_fit(method.name, M_hat, Y, train_mask, val_mask, test_mask, info)
        rows.append(row)
        print(row)

    results = pd.DataFrame(rows).sort_values("test_rmse")
    os.makedirs(args.out_dir, exist_ok=True)
    results.to_csv(os.path.join(args.out_dir, "gdsc_realdata_results.csv"), index=False)

    pd.DataFrame({"row_name": row_names}).to_csv(os.path.join(args.out_dir, "row_names.csv"), index=False)
    pd.DataFrame({"col_name": col_names}).to_csv(os.path.join(args.out_dir, "col_names.csv"), index=False)
    pd.DataFrame({"W_feature": W_names}).to_csv(os.path.join(args.out_dir, "W_features.csv"), index=False)
    pd.DataFrame({"Z_feature": Z_names}).to_csv(os.path.join(args.out_dir, "Z_features.csv"), index=False)

    print("\nFinal results sorted by test RMSE:")
    print(results.to_string(index=False))
    print("\nSaved to:", args.out_dir)
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="gdsc_refined_out")
    p.add_argument("--n_rows_keep", type=int, default=800)
    p.add_argument("--n_cols_keep", type=int, default=200)
    p.add_argument("--n_iter", type=int, default=6)
    p.add_argument("--train_frac", type=float, default=0.50, help="Fraction of observed entries used for training. Use smaller values for sparse experiments.")
    p.add_argument("--val_frac", type=float, default=0.20)
    p.add_argument("--rank", type=int, default=10, help="Rank for the nonconvex low-rank baseline only.")
    p.add_argument("--sigma_base", type=float, default=1.0, help="Base noise scale after standardization; validation chooses lambda multiplier.")
    p.add_argument("--c_lambda", type=float, default=1.0)
    p.add_argument("--max_iter", type=int, default=250)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


if __name__ == "__main__":
    run_real_data_experiment(parse_args())
