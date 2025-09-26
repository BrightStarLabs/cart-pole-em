#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-Entropy Method (CEM) to learn *asymmetric per-parameter* uniform ranges
for θ so that random draws are stable (no overflow, bounded) and non-trivial.

θ_j ~ U[μ_j - W_j, μ_j + W_j]  with j in {1,c,l,r,m,cl,cr,cm}
We optimize (μ, W) directly. W kept >= W_min via softplus; bounds enforced by clipping.

Faster than naive search:
 - Batched RK4 across J initializations
 - Early bailout if any trajectory explodes
 - Elite quantile updates with smoothing
"""

import numpy as np
import random, time, json, argparse, os
import matplotlib.pyplot as plt

# ---------------- Config ----------------
DEFAULTS = dict(
    N=30, T_eval=150, T=400, dt=0.3, alpha_m=0.1, init_scale=3e-1,
    max_amplitude=10.0, min_rms=0.05, burn_in=50, check_every=5,
    pop=72, elite_frac=0.2, iters=40, smooth_mu=0.6, smooth_w=0.6,
    seed=123, verify_plot=True, viz=True, viz_samples=8, viz_vmax=2.0,
    out_dir="viz_out", fast=True,  # fast=True: trims T_eval a bit
    W_min=0.01,  # minimum half-width to avoid collapse to delta
)

PARAMS = ["1","c","l","r","m","cl","cr","cm"]

# ---------- Dynamics (batched) ----------
def deriv_batch(a, m, t, alpha_m):
    l = np.roll(a, 1, axis=1)
    r = np.roll(a, -1, axis=1)
    c = a
    dot_a = (t["theta1"]
             + t["theta_c"]*c + t["theta_l"]*l + t["theta_r"]*r + t["theta_m"]*m
             + t["theta_cl"]*c*l + t["theta_cr"]*c*r + t["theta_cm"]*c*m) - 5.0*alpha_m*c
    dot_m = alpha_m * (a - m)
    return dot_a, dot_m

def rk4_step_batch(a, m, dt, theta, alpha_m):
    k1_a, k1_m = deriv_batch(a, m, theta, alpha_m)
    k2_a, k2_m = deriv_batch(a + 0.5*dt*k1_a, m + 0.5*dt*k1_m, theta, alpha_m)
    k3_a, k3_m = deriv_batch(a + 0.5*dt*k2_a, m + 0.5*dt*k2_m, theta, alpha_m)
    k4_a, k4_m = deriv_batch(a + dt*k3_a,     m + dt*k3_m,     theta, alpha_m)
    a_next = a + (dt/6.0)*(k1_a + 2*k2_a + 2*k3_a + k4_a)
    m_next = m + (dt/6.0)*(k1_m + 2*k2_m + 2*k3_m + k4_m)
    return a_next, m_next

# ---------- Range utils ----------
SUPER = {
    "1":  (-0.05, 0.05),    # keep bias near 0
    "c":  (-0.9, 0.9),
    "l":  (-0.9, 0.9),
    "r":  (-0.9, 0.9),
    "m":  (-0.6, 0.6),
    "cl": (-0.35,0.35),
    "cr": (-0.35,0.35),
    "cm": (-0.6, 0.2)       # gentle negative preferred
}

def clip_muW(mu, W, p):
    lo, hi = SUPER[p]
    W = np.maximum(W, 0.0)
    # keep interval within super-bounds
    W = np.minimum(W, np.minimum(mu - lo, hi - mu))
    # if degenerate, pull to center with tiny width
    if W <= 1e-6:
        mu = 0.5*(lo+hi)
        W = min(0.05, (hi-lo)/4.0)
    return mu, W

def theta_from_muW(rng, MU, W):
    U = rng.uniform
    return {
        "theta1":  U(MU["1"] - W["1"],  MU["1"] + W["1"]),
        "theta_c": U(MU["c"] - W["c"],  MU["c"] + W["c"]),
        "theta_l": U(MU["l"] - W["l"],  MU["l"] + W["l"]),
        "theta_r": U(MU["r"] - W["r"],  MU["r"] + W["r"]),
        "theta_m": U(MU["m"] - W["m"],  MU["m"] + W["m"]),
        "theta_cl":U(MU["cl"]- W["cl"], MU["cl"]+ W["cl"]),
        "theta_cr":U(MU["cr"]- W["cr"], MU["cr"]+ W["cr"]),
        "theta_cm":U(MU["cm"]- W["cm"], MU["cm"]+ W["cm"]),
    }

# ---------- Scoring ----------
def eval_theta_batched(theta, init_a, init_m, T, dt, alpha_m,
                       max_amp, min_rms, burn_in, check_every):
    a = init_a.copy()
    m = init_m.copy()
    J, N = a.shape
    alive = np.ones(J, dtype=bool)
    rms_accum = np.zeros(J, float); rms_count = 0

    for t in range(1, T+1):
        a, m = rk4_step_batch(a, m, dt, theta, alpha_m)

        if (t % check_every) == 0 or t == T:
            ok_finite = np.isfinite(a).all(axis=1) & np.isfinite(m).all(axis=1)
            maxabs = np.maximum(np.max(np.abs(a), axis=1), np.max(np.abs(m), axis=1))
            ok_amp = (maxabs <= max_amp)
            alive &= ok_finite & ok_amp
            if not alive.any():
                return 0.0

        if t >= burn_in:
            rms_t = np.sqrt(np.mean(a*a, axis=1))
            rms_accum += rms_t * alive
            rms_count += 1

    if rms_count == 0:
        return 0.0
    mean_rms = np.zeros_like(rms_accum)
    mean_rms[alive] = rms_accum[alive] / float(rms_count)
    passed = alive & (mean_rms >= min_rms)
    return float(np.count_nonzero(passed)) / float(J)

def score_distribution(MU, W, rng_master, init_a, init_m, T_eval, dt, alpha_m,
                       max_amp, min_rms, burn_in, check_every, S_theta=64):
    # Optional linear sanity bonus
    center = (MU["c"] + MU["l"] + MU["r"]) - 5.0*alpha_m
    halfw  = (W["c"] + W["l"] + W["r"])
    lin = max(0.0, 1.0 - (abs(center) + halfw) / 3.0)

    passed = 0.0
    for _ in range(S_theta):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31-1)))
        theta = theta_from_muW(rng, MU, W)
        try:
            frac = eval_theta_batched(theta, init_a, init_m, T_eval, dt, alpha_m,
                                      max_amp, min_rms, burn_in, check_every)
        except FloatingPointError:
            frac = 0.0
        passed += frac
    pass_rate = passed / float(S_theta)
    # Blend (mostly pass rate)
    return 0.9*pass_rate + 0.1*lin, pass_rate, lin

# ---------- CEM loop ----------
def cem_opt(args):
    np.seterr(over='raise', invalid='raise')
    random.seed(args.seed)
    master = np.random.default_rng(args.seed)

    if args.fast:
        args.T_eval = max(150, min(args.T_eval, args.T//2))

    # Fixed init batch for comparability
    rng_inits = np.random.default_rng(0xBEEF)
    init_a = args.init_scale * rng_inits.standard_normal((8, args.N))
    init_m = args.init_scale * rng_inits.standard_normal((8, args.N))

    # Initialize μ, W near conservative priors
    MU = {p: 0.0 for p in PARAMS}
    MU["cm"] = -0.1
    W  = {"1":0.0, "c":0.2, "l":0.2, "r":0.2, "m":0.15, "cl":0.06, "cr":0.06, "cm":0.08}
    # Clip to supers
    for p in PARAMS:
        MU[p], W[p] = clip_muW(MU[p], W[p], p)
        W[p] = max(W[p], args.W_min) if p != "1" else 0.0

    pop = args.pop
    elite_k = max(1, int(np.ceil(args.elite_frac * pop)))

    best = None; best_score=-1; best_pass=-1; best_MU=None; best_W=None
    t0 = time.time()
    for it in range(1, args.iters+1):
        # Sample population of distributions by jittering MU and W slightly (CEM variant)
        cand = []
        for i in range(pop):
            # small exploration noise relative to current widths
            MU_i = {}
            W_i  = {}
            for p in PARAMS:
                mu_noise = float(master.normal(0.0, 0.25 * (W[p] if W[p]>0 else 0.05)))
                w_noise  = float(master.normal(0.0, 0.15 * (W[p] if W[p]>0 else 0.05)))
                MU_i[p] = MU[p] + mu_noise
                W_i[p]  = abs(W[p] + w_noise)
                MU_i[p], W_i[p] = clip_muW(MU_i[p], W_i[p], p)
                if p != "1":
                    W_i[p] = max(W_i[p], args.W_min)
                else:
                    W_i[p] = 0.0  # keep bias fixed width
            score, pr, lin = score_distribution(MU_i, W_i, master, init_a, init_m,
                                                args.T_eval, args.dt, args.alpha_m,
                                                args.max_amplitude, args.min_rms,
                                                args.burn_in, args.check_every,
                                                S_theta=32 if args.fast else 64)
            cand.append((score, pr, lin, MU_i, W_i))

        cand.sort(key=lambda x: x[0], reverse=True)
        elites = cand[:elite_k]

        # Track best
        if elites[0][0] > best_score:
            best_score, best_pass = elites[0][0], elites[0][1]
            best_MU, best_W = elites[0][3], elites[0][4]

        # Update MU, W toward elite means (with smoothing)
        MU_new = {p: 0.0 for p in PARAMS}
        W_new  = {p: 0.0 for p in PARAMS}
        for p in PARAMS:
            MU_mean = np.mean([e[3][p] for e in elites])
            W_mean  = np.mean([e[4][p] for e in elites])
            MU_new[p] = args.smooth_mu*MU[p] + (1-args.smooth_mu)*MU_mean
            W_new[p]  = args.smooth_w *W[p]  + (1-args.smooth_w) *W_mean
            MU_new[p], W_new[p] = clip_muW(MU_new[p], W_new[p], p)
            if p != "1":
                W_new[p] = max(W_new[p], args.W_min)
            else:
                W_new[p] = 0.0

        MU, W = MU_new, W_new

        if args.fast:
            print(f"[{it:02d}] best_score={best_score:.3f} pass={best_pass:.3f} | "
                  f"MU_c={MU['c']:.3f} W_c={W['c']:.3f} | MU_cm={MU['cm']:.3f} W_cm={W['cm']:.3f}")
        else:
            print(f"[{it:02d}] best_score={best_score:.3f} pass={best_pass:.3f}")

        # Early stop if we’re already very good
        if best_pass >= 0.98:
            break

    elapsed = time.time() - t0
    print("\n=== BEST DISTRIBUTION (μ, W) ===")
    pretty = {k: round(v,5) for k,v in (best_MU|best_W).items()} if hasattr(dict,'|') else {**{('mu_'+k):v for k,v in best_MU.items()}, **{('W_'+k):v for k,v in best_W.items()}}
    print(json.dumps(pretty, indent=2))
    print(f"time={elapsed:.2f}s  best_pass_rate={best_pass:.3f}  best_score={best_score:.3f}")
    return best_MU, best_W

# ---------- Viz ----------
def ensure_outdir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def verify_plot(MU, W, args, seed=2024):
    rng = np.random.default_rng(seed)
    theta = theta_from_muW(rng, MU, W)
    a = args.init_scale * rng.standard_normal(args.N)
    m = args.init_scale * rng.standard_normal(args.N)
    A = np.empty((args.T+1, args.N)); A[0]=a
    for t in range(1, args.T+1):
        aa, mm = rk4_step_batch(a[None,:], m[None,:], args.dt, theta, args.alpha_m)
        a, m = aa[0], mm[0]; A[t]=a
    ensure_outdir(args.out_dir)
    fn = os.path.join(args.out_dir, "verify_best_distribution.png")
    plt.figure(figsize=(8,6))
    plt.imshow(A, origin="upper", aspect="auto", interpolation="nearest",
               vmin=-args.viz_vmax, vmax=args.viz_vmax)
    plt.title("a(x,t) — sampled θ from learned (μ,W)")
    plt.colorbar(label="a"); plt.xlabel("space i"); plt.ylabel("time")
    plt.tight_layout(); plt.savefig(fn, dpi=130); plt.close()
    print(f"Saved: {fn}")

def spacetime_grid(MU, W, args, K=None):
    ensure_outdir(args.out_dir)
    if K is None: K = args.viz_samples
    cols = min(4, K); rows = int(np.ceil(K/cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), squeeze=False)
    for k in range(K):
        rng = np.random.default_rng(100 + k)
        theta = theta_from_muW(rng, MU, W)
        a = args.init_scale * rng.standard_normal(args.N)
        m = args.init_scale * rng.standard_normal(args.N)
        A = np.empty((args.T+1, args.N)); A[0]=a
        alive = True
        for t in range(1, args.T+1):
            aa, mm = rk4_step_batch(a[None,:], m[None,:], args.dt, theta, args.alpha_m)
            a, m = aa[0], mm[0]; A[t]=a
            if (t % args.check_every)==0 or t==args.T:
                if not (np.isfinite(a).all() and np.isfinite(m).all()): alive=False; break
                if max(np.max(np.abs(a)), np.max(np.abs(m))) > args.max_amplitude: alive=False; break
        r, c = k//cols, k%cols
        ax = axs[r,c]
        im = ax.imshow(A if alive else np.zeros_like(A), origin="upper", aspect="auto",
                       interpolation="nearest", vmin=-args.viz_vmax, vmax=args.viz_vmax)
        ax.set_title(f"sample {k+1} — {'OK' if alive else 'FAIL'}", fontsize=10)
        ax.set_xlabel("space i"); ax.set_ylabel("time")
    plt.tight_layout()
    fig.colorbar(im, ax=axs, shrink=0.6, label="a")
    fn = os.path.join(args.out_dir, "spacetime_grid.png")
    plt.savefig(fn, dpi=130); plt.close(fig)
    print(f"Saved: {fn}")

# ---------- CLI ----------
def build_argparser():
    p = argparse.ArgumentParser(description="CEM over (μ,W) for θ ranges (fast, asymmetric).")
    p.add_argument("--N", type=int, default=DEFAULTS["N"])
    p.add_argument("--T_eval", type=int, default=DEFAULTS["T_eval"])
    p.add_argument("--T", type=int, default=DEFAULTS["T"])
    p.add_argument("--dt", type=float, default=DEFAULTS["dt"])
    p.add_argument("--alpha_m", type=float, default=DEFAULTS["alpha_m"])
    p.add_argument("--init_scale", type=float, default=DEFAULTS["init_scale"])
    p.add_argument("--max_amplitude", type=float, default=DEFAULTS["max_amplitude"])
    p.add_argument("--min_rms", type=float, default=DEFAULTS["min_rms"])
    p.add_argument("--burn_in", type=int, default=DEFAULTS["burn_in"])
    p.add_argument("--check_every", type=int, default=DEFAULTS["check_every"])
    p.add_argument("--pop", type=int, default=DEFAULTS["pop"])
    p.add_argument("--elite_frac", type=float, default=DEFAULTS["elite_frac"])
    p.add_argument("--iters", type=int, default=DEFAULTS["iters"])
    p.add_argument("--smooth_mu", type=float, default=DEFAULTS["smooth_mu"])
    p.add_argument("--smooth_w", type=float, default=DEFAULTS["smooth_w"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--fast", action="store_true" if DEFAULTS["fast"] else "store_false")
    p.add_argument("--verify_plot", action="store_true" if DEFAULTS["verify_plot"] else "store_false")
    p.add_argument("--viz", action="store_true" if DEFAULTS["viz"] else "store_false")
    p.add_argument("--viz_samples", type=int, default=DEFAULTS["viz_samples"])
    p.add_argument("--viz_vmax", type=float, default=DEFAULTS["viz_vmax"])
    p.add_argument("--out_dir", type=str, default=DEFAULTS["out_dir"])
    p.add_argument("--W_min", type=float, default=DEFAULTS["W_min"])
    return p

def main():
    args = build_argparser().parse_args()
    MU, W = cem_opt(args)

    # --- After MU, W are learned ---
    print("\n# --- Paste into your θ init (NumPy version) ---")
    print("theta = {")
    for lab, name in [("1","theta1"),("c","theta_c"),("l","theta_l"),("r","theta_r"),
                      ("m","theta_m"),("cl","theta_cl"),("cr","theta_cr"),("cm","theta_cm")]:
        lo = MU[lab] - W[lab]; hi = MU[lab] + W[lab]
        if name == "theta1":
            print("    'theta1': 0.0,")
        else:
            print(f"    '{name}': np.random.uniform({lo:.16f}, {hi:.16f}),")
    print("}")

    # --- Your requested random.uniform + self.theta block ---
    import random
    print("\n# --- Paste this directly into your class (random.uniform) ---")
    print("self.theta = {")
    # theta1 fixed to 0, as requested
    print('            "theta1": 0,')
    # per-parameter learned ranges
    lo_c,  hi_c  = MU['c']  - W['c'],  MU['c']  + W['c']
    lo_l,  hi_l  = MU['l']  - W['l'],  MU['l']  + W['l']
    lo_r,  hi_r  = MU['r']  - W['r'],  MU['r']  + W['r']
    lo_m,  hi_m  = MU['m']  - W['m'],  MU['m']  + W['m']
    lo_cl, hi_cl = MU['cl'] - W['cl'], MU['cl'] + W['cl']
    lo_cr, hi_cr = MU['cr'] - W['cr'], MU['cr'] + W['cr']
    lo_cm, hi_cm = MU['cm'] - W['cm'], MU['cm'] + W['cm']

    print(f'            "theta_c": random.uniform({lo_c:.16f},{hi_c:.16f}),')
    print(f'            "theta_l":  random.uniform({lo_l:.16f},{hi_l:.16f}),')
    print(f'            "theta_r":  random.uniform({lo_r:.16f},{hi_r:.16f}),')
    print(f'            "theta_m":  random.uniform({lo_m:.16f},{hi_m:.16f}),')
    print(f'            "theta_cl": random.uniform({lo_cl:.16f},{hi_cl:.16f}),')
    print(f'            "theta_cr": random.uniform({lo_cr:.16f},{hi_cr:.16f}),')
    print(f'            "theta_cm": random.uniform({lo_cm:.16f},{hi_cm:.16f}),  # small negative helps soft saturation')
    print("        }")


    if args.verify_plot:
        verify_plot(MU, W, args)
    if args.viz:
        spacetime_grid(MU, W, args)

if __name__ == "__main__":
    main()
