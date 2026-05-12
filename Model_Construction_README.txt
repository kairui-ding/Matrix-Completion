compare2.py
===================

This is a refined version of compare1.py. It separates two different uses:

1) Theorem study
   Use --study theorem.
   This uses only our one-stage estimator, all observed entries for fitting, and a fixed theorem-style lambda:
       lambda = C_lambda * sigma * sqrt(max(n1,n2) * p_fit).
   No validation tuning is done.
   It reports component errors for A, B, C, L and infinity-type errors.

2) Method-comparison study
   Use --study comparison.
   This uses train/validation/test splitting and tunes lambda for each method using noisy validation observations.
   It compares:
       cov_interaction_only
       common_lowrank_convex
       paper1_lowrank_nonconvex
       paper2_additive_no_interaction
       ours_full_interaction_latent

Examples
--------

Quick theorem check:
    python compare1_refined.py --study theorem --mode vary_p --quick --out_dir out_theory_quick

Main theorem checks:
    python compare1_refined.py --study theorem --mode vary_p --repeats 10 --out_dir out_theory_p
    python compare1_refined.py --study theorem --mode vary_sigma --repeats 10 --out_dir out_theory_sigma
    python compare1_refined.py --study theorem --mode vary_n --repeats 8 --out_dir out_theory_n

Quick method comparison:
    python compare1_refined.py --study comparison --mode vary_p --quick --out_dir out_cmp_quick

Main method comparisons:
    python compare1_refined.py --study comparison --mode vary_p --repeats 5 --out_dir out_cmp_p
    python compare1_refined.py --study comparison --mode vary_interaction --repeats 5 --out_dir out_cmp_interaction
    python compare1_refined.py --study comparison --mode vary_latent --repeats 5 --out_dir out_cmp_latent

Important differences from the original compare1.py
---------------------------------------------------

- For method comparison, validation tuning uses noisy held-out observations Y_obs, not clean M_true.
- For theorem checks, lambda is fixed and not selected by validation.
- The theorem study uses all observed entries for fitting, so p is the estimator's actual sampling rate.
- Component-level errors for A, B, C, L are reported in theorem mode.
- The strong covariate baseline is renamed cov_interaction_only because it contains W A Z^T.
- The low-rank convex baseline is included.
- Both noisy held-out RMSE and clean-signal metrics are reported in synthetic method-comparison mode.
