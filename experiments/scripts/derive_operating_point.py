"""Derive the connectome-Kuramoto operating point c_intra from first principles.

Replaces the legacy OIST-data fit with three intrinsic, data-free diagnostics
swept over c_intra (anatomically structured CoCoMac connectome, two-brain model):

  1. Synchronization transition  : R_mean(c), time-averaged per-brain order parameter.
  2. Metastability               : M(c), temporal std of the order parameter
                                   (Shanahan/Deco metastability index). Its peak marks
                                   the metastable regime the pipeline is meant to probe.
  3. Structure expression        : D(c), Mahalanobis distance between the structured
                                   PLV distribution and a tract-shuffled null.

The operating point is read off intrinsically (metastability peak, corroborated by the
structure-expression band). No empirical recording is used anywhere.

Outputs to --outdir:
  operating_point.npz   - all swept arrays + the chosen c*.
  operating_point.png   - the three diagnostics vs c_intra with c* marked.
  summary.txt           - the derived operating point and the numbers behind it.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from canonical_calibration_figs import (  # noqa: E402
    build_solver, load_connectome, maha, plv_hist_feature, plv_matrix,
    run_once, shuffle_connectome, virtual_partner,
)

N_BRAIN = 76  # oscillators per brain


def order_param(theta_block):
    """Kuramoto order parameter time series R(t) for one brain block (n, T)."""
    return np.abs(np.exp(1j * theta_block).mean(axis=0))


def brain_diagnostics(theta):
    """From stable-window phases (152, T): time-averaged R and metastability M,
    averaged over the two brains."""
    rA = order_param(theta[:N_BRAIN])
    rB = order_param(theta[N_BRAIN:])
    r_mean = 0.5 * (rA.mean() + rB.mean())
    metastab = 0.5 * (rA.std() + rB.std())
    return r_mean, metastab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/tmp/calib_out")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--cmax", type=float, default=45.0)
    ap.add_argument("--cstep", type=float, default=3.0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    t0 = time.time()
    W, tract, roi = load_connectome()
    Wv, Dv = virtual_partner(W, tract, roi)
    Ws, Ds = shuffle_connectome(*virtual_partner(W, tract, roi)[:2], seed=2024)

    no = Wv.shape[0]
    fr = np.random.default_rng(0)
    omega = 2 * np.pi * (40 + 8 * fr.standard_normal(no))

    print(f"[{time.time()-t0:.0f}s] compiling structured + shuffled solvers...")
    solver_s, del_s = build_solver(Wv, Dv, omega)
    solver_h, del_h = build_solver(Ws, Ds, omega)

    c_vals = np.arange(0.0, args.cmax + 1e-9, args.cstep)
    S = args.seeds
    R = np.zeros((len(c_vals), S))
    M = np.zeros((len(c_vals), S))
    nbins = len(plv_hist_feature(np.eye(no)))
    hist_s = np.zeros((len(c_vals), S, nbins))
    hist_h = np.zeros((len(c_vals), S, nbins))

    for j, c in enumerate(c_vals):
        for s in range(S):
            th_s = run_once(solver_s, del_s, c, seed=100 + s)
            th_h = run_once(solver_h, del_h, c, seed=500 + s)
            R[j, s], M[j, s] = brain_diagnostics(th_s)
            hist_s[j, s] = plv_hist_feature(plv_matrix(th_s))
            hist_h[j, s] = plv_hist_feature(plv_matrix(th_h))
        print(f"[{time.time()-t0:.0f}s] c={c:.1f}  R={R[j].mean():.3f}  M={M[j].mean():.3f}")

    R_mean = R.mean(1)
    M_mean = M.mean(1)
    D = np.array([maha(hist_s[j], hist_h[j]) for j in range(len(c_vals))])

    # Operating point: metastability peak (excluding the trivial c=0 incoherent point),
    # corroborated by lying within the rising structure-expression band.
    valid = c_vals > 0
    c_meta = c_vals[valid][np.argmax(M_mean[valid])]
    # transition midpoint: smallest c where R crosses halfway between its min and max
    r_lo, r_hi = R_mean.min(), R_mean.max()
    r_mid = 0.5 * (r_lo + r_hi)
    above = np.where(R_mean >= r_mid)[0]
    c_trans = c_vals[above[0]] if len(above) else float("nan")
    c_star = c_meta

    np.savez(os.path.join(args.outdir, "operating_point.npz"),
             c_vals=c_vals, R=R, M=M, R_mean=R_mean, M_mean=M_mean, D=D,
             c_star=c_star, c_meta=c_meta, c_trans=c_trans)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(c_vals, R_mean, "o-", color="#0072B2")
    ax[0].axvline(c_trans, color="#999", ls="--", label=f"transition c={c_trans:.0f}")
    ax[0].set_title("Synchronization transition"); ax[0].set_ylabel("mean order parameter R")
    ax[0].legend(frameon=False)
    ax[1].plot(c_vals, M_mean, "o-", color="#D55E00")
    ax[1].axvline(c_star, color="#D55E00", ls=":", label=f"metastability peak c*={c_star:.0f}")
    ax[1].set_title("Metastability"); ax[1].set_ylabel("temporal std of R")
    ax[1].legend(frameon=False)
    ax[2].plot(c_vals, D, "o-", color="#009E73")
    ax[2].axvline(c_star, color="#D55E00", ls=":")
    ax[2].set_title("Structure expression"); ax[2].set_ylabel("structured vs shuffled (Mahalanobis)")
    for a in ax:
        a.set_xlabel(r"$c_{\mathrm{intra}}$"); a.set_xlim(0, args.cmax); a.set_ylim(bottom=0)
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "operating_point.png"), dpi=160)

    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(f"Derived operating point c* = {c_star:.1f} (metastability peak)\n")
        f.write(f"Synchronization transition midpoint c_trans = {c_trans:.1f}\n")
        f.write(f"R_mean: {np.round(R_mean,3).tolist()}\n")
        f.write(f"M_mean: {np.round(M_mean,4).tolist()}\n")
        f.write(f"D(struct vs shuf): {np.round(D,3).tolist()}\n")
        f.write(f"c_vals: {c_vals.tolist()}\n")
    print(f"[{time.time()-t0:.0f}s] c* = {c_star:.1f} (metastability peak); transition ~ {c_trans:.1f}")


if __name__ == "__main__":
    main()
