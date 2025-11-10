#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi_sample_v8a.py — fast calibration + controllable EM workload
"""
from __future__ import annotations
import argparse, sys, os, json, math
from typing import List, Dict, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def mad(x):
    med = np.nanmedian(x)
    return np.nanmedian(np.abs(x - med)), med

def cap_weights(w, pct=99.0):
    thr = np.nanpercentile(w, pct)
    return np.minimum(w, thr)

def simulate_genotypes(num_samples: int = 5,
                       num_snps: int = 300_000,
                       het_rate_max: float = 0.05,
                       rng: np.random.Generator | None = None):
    rng = np.random.default_rng() if rng is None else rng
    G = np.zeros((num_samples, num_snps), dtype=np.float32)
    het_rates = rng.uniform(0.0, het_rate_max, size=num_samples)
    for s in range(num_samples):
        homo = rng.integers(0, 2, size=num_snps).astype(np.float32)
        het_mask = rng.random(num_snps) < het_rates[s]
        g = homo
        g[het_mask] = 0.5
        G[s] = g
    return G, het_rates

def simulate_mixture_weights(num_samples: int = 5, rng: np.random.Generator | None = None):
    rng = np.random.default_rng() if rng is None else rng
    return rng.dirichlet(np.ones(num_samples, dtype=np.float64))

def simulate_depth_allocation(num_snps: int, w: np.ndarray, total_depth_mean: float,
                              rng: np.random.Generator | None = None):
    rng = np.random.default_rng() if rng is None else rng
    n_total = rng.poisson(total_depth_mean, size=num_snps).astype(np.int32)
    n_total[n_total < 1] = 1
    n_by_sample = np.zeros((len(w), num_snps), dtype=np.int32)
    for i in range(num_snps):
        n_by_sample[:, i] = rng.multinomial(n_total[i], w)
    return n_total, n_by_sample

def simulate_allele_counts(G: np.ndarray, n_by_sample: np.ndarray, e: float,
                           rng: np.random.Generator | None = None):
    rng = np.random.default_rng() if rng is None else rng
    S, N = G.shape
    k_by_sample = np.zeros((S, N), dtype=np.int32)
    p = np.where(G == 1.0, 1.0 - e, np.where(G == 0.0, e, 0.5)).astype(np.float32)
    for s in range(S):
        k_by_sample[s] = rng.binomial(n_by_sample[s], p[s])
    k_total = k_by_sample.sum(axis=0).astype(np.int32)
    n_total = n_by_sample.sum(axis=0).astype(np.int32)
    return k_by_sample, k_total, n_total

def error_correct_fraction_unclipped(k_total: np.ndarray, n_total: np.ndarray, e: float):
    with np.errstate(divide='ignore', invalid='ignore'):
        p_hat = k_total / n_total
        c = (p_hat - e) / (1.0 - 2.0 * e)
    c[np.isnan(c)] = np.nan
    return c

def mask_and_weights(c, g_std, n_total, min_n=5, zmax=5.0):
    m = (~np.isnan(c)) & (~np.isnan(g_std)) & (n_total >= min_n) & ((g_std == 0.0) | (g_std == 1.0))
    c1 = c[m]; g1 = g_std[m]; wts = n_total[m].astype(float)
    wts = cap_weights(wts, 99.0)
    madv, med = mad(c1[~np.isnan(c1)])
    if np.isfinite(madv) and madv > 0:
        z = np.abs(c1 - med) / (1.4826 * madv + 1e-12)
        keep = z <= zmax
        c1 = c1[keep]; g1 = g1[keep]; wts = wts[keep]
    return c1, g1, wts

def weighted_cov(x, y, w):
    xm = np.average(x, weights=w); ym = np.average(y, weights=w)
    return float(np.average((x - xm) * (y - ym), weights=w))

def weighted_var(x, w):
    xm = np.average(x, weights=w)
    return float(np.average((x - xm) ** 2, weights=w))

def estimate_w_by_covariance(c_unclipped, g_std, n_total, min_n=5):
    c1, g1, wts = mask_and_weights(c_unclipped, g_std, n_total, min_n=min_n)
    if len(c1) < 100: return float('nan')
    cw = weighted_cov(g1, c1, wts); vw = weighted_var(g1, wts)
    if vw <= 0: return float('nan')
    return float(np.clip(cw / vw, 0.0, 1.0))

def safe_weighted_sse(resid, wts, delta=0.05):
    r = np.nan_to_num(resid, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    r = np.clip(r, -1.0, 1.0)
    w = np.nan_to_num(wts,   nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    abs_r = np.abs(r)
    quad = 0.5 * (abs_r**2)
    lin = delta * (abs_r - 0.5*delta)
    loss = np.where(abs_r <= delta, quad, lin)
    return float(np.sum(loss * w))

def em_k_backgrounds_auto(c, g, n, K=4, max_iter=200, min_iter=24,
                          tau_schedule=(1.0,0.7,0.5,0.3,0.2,0.1,0.05),
                          n_init=7, seed=0, init_w=None, tol=1e-6,
                          w_prior=None, w_prior_tau=0.07,
                          min_sep=0.02):
    rng = np.random.default_rng(seed)
    c = c.astype(float); g = g.astype(float); wts = n.astype(float)
    if len(c) < 1000:
        return dict(success=False, reason="too_few_loci", K=K)
    def single_run(seed_j):
        rngj = np.random.default_rng(seed_j)
        w0 = estimate_w_by_covariance(c, g, wts) if init_w is None else init_w
        if not (w0 == w0): w0 = 0.5
        if w_prior is not None: w0 = 0.7*w0 + 0.3*w_prior
        w = float(np.clip(w0, 1e-6, 1.0-1e-6))
        t0 = (c - w*g) / max(1.0 - w, 1e-8)
        qs = np.linspace(0.05, 0.95, K)
        f = np.quantile(t0, qs).astype(float) + rngj.normal(0, 0.01, size=K)
        f = np.clip(f, 0.0, 1.0); f.sort()
        for _ in range(3):
            for k in range(1, K):
                if f[k] - f[k-1] < min_sep:
                    f[k] = min(1.0, f[k-1] + min_sep)
        pi = np.ones(K)/K
        prev_obj = None; iters_total = 0
        steps_per_tau = max(5, max_iter // len(tau_schedule))
        for tau in tau_schedule:
            for _ in range(steps_per_tau):
                t = (c - w*g) / max(1.0 - w, 1e-8)
                log_r = np.empty((K, len(t)), dtype=float)
                for k in range(K):
                    diff = (t - f[k])
                    log_r[k] = np.log(pi[k] + 1e-12) - (wts * diff*diff) / (2.0 * tau)
                a = log_r.max(axis=0); r = np.exp(log_r - a); r /= np.sum(r, axis=0, keepdims=True)
                Nk = np.sum(r * wts, axis=1) + 1e-12
                f = np.sum(r * (wts * t), axis=1) / Nk
                f = np.clip(f, 0.0, 1.0); f.sort()
                for k in range(1, K):
                    if f[k] - f[k-1] < min_sep:
                        f[k] = min(1.0, f[k-1] + min_sep)
                pi = Nk / np.sum(Nk)
                fbar = r.T @ f
                y = c - fbar; z = g - fbar
                z_mean = np.average(z, weights=wts); y_mean = np.average(y, weights=wts)
                cov = np.average((z - z_mean)*(y - y_mean), weights=wts)
                var = np.average((z - z_mean)**2, weights=wts)
                if var > 0:
                    w_mle = float(np.clip(cov/var, 0.0, 1.0))
                    if w_prior is not None:
                        w = float(np.clip((w_mle + (var * w_prior_tau**-2) * w_prior) / (1.0 + (var * w_prior_tau**-2)), 0.0, 1.0))
                    else:
                        w = w_mle
                pred = w*g + (1.0 - w)*fbar
                resid = c - pred
                obj = safe_weighted_sse(resid, wts, delta=0.05)
                if w_prior is not None:
                    obj += ((w - w_prior)**2) / (2.0 * w_prior_tau**2)
                iters_total += 1
                if (prev_obj is not None) and (iters_total >= min_iter):
                    if abs(prev_obj - obj) <= tol * max(1.0, prev_obj):
                        return dict(success=True, K=K, w=float(w), f=[float(x) for x in f], pi=[float(x) for x in pi],
                                    sse=float(obj), n_loci=int(len(c)), iters=int(iters_total))
                prev_obj = obj
        return dict(success=True, K=K, w=float(w), f=[float(x) for x in f], pi=[float(x) for x in pi],
                    sse=float(prev_obj if prev_obj is not None else obj), n_loci=int(len(c)),
                    iters=int(iters_total))
    best = None
    for j in range(n_init):
        res = single_run(seed + j * 97 + 13)
        if (res is not None) and isinstance(res, dict):
            if best is None or (res.get("success") and res.get("sse", float('inf')) < best.get("sse", float('inf'))):
                best = res
    if best is None:
        return dict(success=False, reason="init_failed_none", K=K)
    return best

def model_select_k_em(c, g, n, K_list=(3,4,5), seed=0, use_bic=False, use_icl=False,
                      w_prior=None, w_prior_tau=0.07, min_sep=0.02, n_init=7,
                      tau_schedule=(1.0,0.7,0.5,0.3,0.2,0.1,0.05), max_iter=220, min_iter=24):
    candidates = []
    for i, K in enumerate(K_list):
        res = em_k_backgrounds_auto(c, g, n, K=K, max_iter=max_iter, min_iter=min_iter,
                                    tau_schedule=tau_schedule, n_init=n_init, seed=seed + 1337*i, init_w=None, tol=1e-6,
                                    w_prior=w_prior, w_prior_tau=w_prior_tau, min_sep=min_sep)
        if (res is None) or (not isinstance(res, dict)):
            res = dict(success=False, reason="none_returned", K=K)
        sse = float(res.get('sse', float('inf')))
        N = int(res.get('n_loci', 1))
        crit = sse
        if use_bic and res.get('success', False):
            p = (K - 1) + K + 1
            bic = N * math.log(max(sse/max(N,1), 1e-12)) + p * math.log(max(N,1))
            crit = float(bic)
            res['bic'] = float(bic)
        if use_icl and res.get('success', False):
            Kcur = K
            w = res.get('w', 0.5); f = np.array(res.get('f', np.linspace(0.1,0.9,Kcur)))
            t = (c - w*g) / max(1.0 - w, 1e-8)
            temp = 0.2
            log_r = np.empty((Kcur, len(t)), dtype=float)
            wts = n.astype(float)
            for k in range(Kcur):
                diff = (t - f[k])
                log_r[k] = -(wts * diff*diff) / (2.0 * temp)
            a = log_r.max(axis=0); r = np.exp(log_r - a); r /= np.sum(r, axis=0, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                ent = -np.sum(wts * np.sum(r * np.log(r + 1e-12), axis=0)) / max(np.sum(wts), 1e-12)
            icl = crit + N * ent
            res['icl'] = float(icl)
            crit = float(icl)
        res['criterion'] = crit
        res['w'] = float(res.get('w', float('nan'))) if 'w' in res else float('nan')
        res['sse'] = sse
        candidates.append(res)
    ok = [r for r in candidates if r.get('success', False)]
    if not ok:
        return dict(success=False, reason="all_failed", tried=candidates)
    best = min(ok, key=lambda r: r['criterion'])
    best_summary = dict(best)
    best_summary['candidates'] = [dict(K=r.get('K'), w=r.get('w'), sse=r.get('sse'),
                                       bic=r.get('bic', None), icl=r.get('icl', None),
                                       criterion=r.get('criterion')) for r in candidates]
    return best_summary

def run_once(num_snps=300_000, e=0.01, depth=20.0, seed=20251106,
             het_max=0.05, std_index=0, kmin=3, kmax=9, use_bic=False, use_icl=False,
             min_n=5, true_k=5, do_em=True,
             em_n_init=7, em_tau=(1.0,0.7,0.5,0.3,0.2,0.1,0.05), em_max_iter=220, em_min_iter=24):
    rng = np.random.default_rng(seed)
    S = 5
    G, het_rates = simulate_genotypes(S, num_snps, het_max, rng)
    g_std = G[std_index].astype(np.float32)
    w_vec = simulate_mixture_weights(S, rng)
    w_true = float(w_vec[std_index])
    n_total, n_by_sample = simulate_depth_allocation(num_snps, w_vec, depth, rng)
    _, k_total, n_total2 = simulate_allele_counts(G, n_by_sample, e, rng)
    assert np.all(n_total == n_total2)
    c_unclip = error_correct_fraction_unclipped(k_total, n_total, e)
    c, g, wts = mask_and_weights(c_unclip, g_std, n_total, min_n=min_n)
    w_cov = estimate_w_by_covariance(c_unclipped=c_unclip, g_std=g_std, n_total=n_total, min_n=min_n)
    result = dict(seed=int(seed), w_true=float(w_true), w_cov=float(w_cov),
                  w_kem=float('nan'), K=-1, kem_success=False)
    if do_em:
        w_prior = None if not (w_cov == w_cov) else float(np.clip(w_cov, 1e-6, 1.0-1.0e-6))
        K_list = list(range(int(kmin), int(kmax)+1))
        em_best = model_select_k_em(c, g, wts, K_list=K_list, seed=seed+17,
                                    use_bic=use_bic, use_icl=use_icl,
                                    w_prior=w_prior, w_prior_tau=0.07, min_sep=0.02,
                                    n_init=em_n_init, tau_schedule=em_tau, max_iter=em_max_iter, min_iter=em_min_iter)
        result.update(w_kem=float(em_best.get('w', float('nan'))) if em_best.get('success', False) else float('nan'),
                      K=int(em_best.get('K', -1)) if em_best.get('success', False) else -1,
                      kem_success=bool(em_best.get('success', False)))
    return result

def fit_affine(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    X = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = beta[0], beta[1]
    return float(a), float(b)
def calibrate_estimators(num_sims, *, num_snps_fast, do_em,
                         calib_kmin, calib_kmax, calib_use_bic, calib_use_icl,
                         calib_n_init, seed, **kwargs):
    """
    Fast calibration by small-SNP sims (and optional light EM).
    We explicitly strip conflicting keys from kwargs to avoid duplicate-arg errors.
    """
    mom_true, mom_est, kem_true, kem_est = [], [], [], []
    base_seed = int(seed)

    # --- make a safe copy of kwargs and drop keys that we will pass explicitly ---
    kwargs2 = dict(kwargs)
    for k in [
        # globals that we will override here
        'seed', 'num_snps',
        # EM / model-selection controls that have *calib_* counterparts
        'kmin', 'kmax', 'use_bic', 'use_icl',
        'em_n_init', 'em_tau', 'em_max_iter', 'em_min_iter',
        # and anything that could collide accidentally in the future
        'do_em'
    ]:
        kwargs2.pop(k, None)

    # --- lighter EM schedule for calibration (or skip EM entirely) ---
    if do_em:
        em_tau = (1.0, 0.5, 0.2)
        em_max_iter = 80
        em_min_iter = 10
    else:
        em_tau = (1.0,)
        em_max_iter = 1
        em_min_iter = 1

    for i in range(num_sims):
        res = run_once(
            seed=base_seed + 123 * i,
            num_snps=num_snps_fast,
            do_em=do_em,
            # use *calib_* knobs during calibration to avoid clashing with kwargs
            kmin=calib_kmin, kmax=calib_kmax,
            use_bic=calib_use_bic, use_icl=calib_use_icl,
            em_n_init=calib_n_init, em_tau=em_tau,
            em_max_iter=em_max_iter, em_min_iter=em_min_iter,
            # safe remainder (depth/e/het_max/std_index/min_n/true_k etc.)
            **kwargs2
        )
        mom_true.append(res['w_true']); mom_est.append(res['w_cov'])
        kem_true.append(res['w_true']); kem_est.append(res['w_kem'])

    a_mom, b_mom = fit_affine(mom_est, mom_true)
    if do_em and np.isfinite(np.nanmean(kem_est)):
        a_kem, b_kem = fit_affine(kem_est, kem_true)
    else:
        a_kem, b_kem = 0.0, 1.0
    return (a_mom, b_mom), (a_kem, b_kem)


def plot_results(runs: List[dict], out_pdf: str, calib=None):
    rep_ids = np.arange(1, len(runs)+1)
    truth = np.array([r['w_true'] for r in runs], dtype=float)
    cov   = np.array([r['w_cov'] for r in runs], dtype=float)
    kem   = np.array([r['w_kem'] for r in runs], dtype=float)
    if calib is not None:
        (a_mom,b_mom),(a_kem,b_kem) = calib
        cov_cal = a_mom + b_mom * cov
        kem_cal = a_kem + b_kem * kem
    else:
        cov_cal = cov; kem_cal = kem
    fig = plt.figure(figsize=(7.5, 7.2), dpi=300)
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(rep_ids, truth, marker='o', linestyle='-', label='Truth')
    ax1.plot(rep_ids, cov,   marker='s', linestyle='--', label='Covariance (raw)')
    ax1.plot(rep_ids, kem,   marker='^', linestyle='--', label='K-EM (raw)')
    if calib is not None:
        ax1.plot(rep_ids, cov_cal, marker='s', linestyle='-', label='Covariance (cal)')
        ax1.plot(rep_ids, kem_cal, marker='^', linestyle='-', label='K-EM (cal)')
    ax1.set_xlabel("Replicate")
    ax1.set_ylabel("w (standard sample proportion)")
    ax1.set_title("Replicate-wise estimates")
    ax1.grid(True, alpha=0.4)
    ax1.legend(ncol=2, fontsize=8)
    ax2 = fig.add_subplot(2,1,2)
    ax2.scatter(truth, cov, marker='s', label='Covariance (raw)')
    ax2.scatter(truth, kem, marker='^', label='K-EM (raw)')
    if calib is not None:
        ax2.scatter(truth, cov_cal, marker='s', facecolors='none', edgecolors='black', label='Covariance (cal)')
        ax2.scatter(truth, kem_cal, marker='^', facecolors='none', edgecolors='black', label='K-EM (cal)')
    lo = 0.0; hi = max(1.0, float(np.nanmax([truth, cov, kem, cov_cal, kem_cal])))
    ax2.plot([lo, hi], [lo, hi], linestyle='-', linewidth=1.2)
    ax2.set_xlabel("Truth")
    ax2.set_ylabel("Estimate")
    ax2.set_title("Parity plot")
    ax2.grid(True, alpha=0.4)
    ax2.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MoM + K-EM with fast calibration and controllable EM workload")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--num-snps", type=int, default=300_000)
    p.add_argument("--depth", type=float, default=20.0)
    p.add_argument("--e", type=float, default=0.01)
    p.add_argument("--het-max", type=float, default=0.05)
    p.add_argument("--std-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=20251106)
    p.add_argument("--true-k", type=int, default=5)
    p.add_argument("--k-min", type=int, default=3)
    p.add_argument("--k-max", type=int, default=9)
    p.add_argument("--use-bic", action="store_true")
    p.add_argument("--use-icl", action="store_true")
    p.add_argument("--min-n", type=int, default=5)
    p.add_argument("--outdir", type=str, default=".")
    p.add_argument("--calib-sims", type=int, default=0)
    p.add_argument("--calib-fast-snps", type=int, default=20000)
    p.add_argument("--calib-use-em", action="store_true")
    p.add_argument("--calib-k-min", type=int, default=3)
    p.add_argument("--calib-k-max", type=int, default=9)
    p.add_argument("--calib-use-bic", action="store_true")
    p.add_argument("--calib-use-icl", action="store_true")
    p.add_argument("--calib-n-init", type=int, default=2)
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    calib = None
    if args.calib_sims and args.calib_sims > 0:
        calib = calibrate_estimators(
            args.calib_sims,
            num_snps_fast=args.calib_fast_snps,
            do_em=args.calib_use_em,
            calib_kmin=args.calib_k_min,
            calib_kmax=args.calib_k_max,
            calib_use_bic=args.calib_use_bic,
            calib_use_icl=args.calib_use_icl,
            calib_n_init=args.calib_n_init,
            seed=args.seed+991,
            num_snps=args.num_snps, e=args.e, depth=args.depth,
            het_max=args.het_max, std_index=args.std_index,
            kmin=args.k_min, kmax=args.k_max, use_bic=args.use_bic, use_icl=args.use_icl,
            min_n=args.min_n, true_k=args.true_k
        )
        (a_mom,b_mom),(a_kem,b_kem) = calib
        print(f"[Calibration] MoM: true ≈ {a_mom:.4f} + {b_mom:.4f} * est ;  K-EM: true ≈ {a_kem:.4f} + {b_kem:.4f} * est (do_em={args.calib_use_em})")
    runs = []
    for i in range(args.runs):
        seed_i = int(args.seed + i*101)
        res = run_once(num_snps=args.num_snps, e=args.e, depth=args.depth, seed=seed_i,
                       het_max=args.het_max, std_index=args.std_index,
                       kmin=args.k_min, kmax=args.k_max, use_bic=args.use_bic, use_icl=args.use_icl,
                       min_n=args.min_n, true_k=args.true_k, do_em=True)
        runs.append(res)
        print(f"[Rep{i+1}] seed={seed_i}  truth={res['w_true']:.6f}  cov={res['w_cov']:.6f}  kem={res['w_kem']:.6f} (K={res['K']})")
    with open(os.path.join(args.outdir, "summary_all.json"), "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)
    lines = ["replicate,seed,w_true,w_cov,w_kem,K,kem_success"]
    for i, r in enumerate(runs, 1):
        lines.append(f"{i},{r['seed']},{r['w_true']},{r['w_cov']},{r['w_kem']},{r['K']},{int(r['kem_success'])}")
    with open(os.path.join(args.outdir, "runs.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    out_pdf = os.path.join(args.outdir, "comparison_v8a.pdf")
    plot_results(runs, out_pdf, calib=calib)
    print("Artifacts written:")
    print(" -", os.path.abspath(os.path.join(args.outdir, "summary_all.json")))
    print(" -", os.path.abspath(os.path.join(args.outdir, "runs.csv")))
    print(" -", os.path.abspath(out_pdf))

if __name__ == "__main__":
    sys.exit(main())
