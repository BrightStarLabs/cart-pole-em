# cma_wrapping_antishift.py
import time
import csv
from typing import Tuple, Optional

import numpy as np
try:
    import cma
except Exception as e:
    raise ImportError("Please install CMA-ES:  pip install cma") from e


# -------------------------
# EM dynamics (3 params), WRAPPING boundary
# -------------------------

def neighbors_wrapped(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (L, C, R) with circular wrap. a: (N,) float32"""
    L = np.roll(a, 1)
    R = np.roll(a, -1)
    return L, a, R

def rhs_linear(a: np.ndarray, K: np.ndarray) -> np.ndarray:
    """da/dt = k1*L + k2*C + k3*R (wrapping)."""
    k1, k2, k3 = K
    L, C, R = neighbors_wrapped(a)
    return k1 * L + k2 * C + k3 * R

def rk4_step(a: np.ndarray, dt: float, K: np.ndarray) -> np.ndarray:
    """One RK4 step for a' = f(a)."""
    s1 = rhs_linear(a, K)
    s2 = rhs_linear(a + 0.5 * dt * s1, K)
    s3 = rhs_linear(a + 0.5 * dt * s2, K)
    s4 = rhs_linear(a + dt * s3, K)
    return a + (dt / 6.0) * (s1 + 2 * s2 + 2 * s3 + s4)


# -------------------------
# Snapshot divergence + anti-shift penalty
# -------------------------

def _circular_corr_max_nonzero(a: np.ndarray, b: np.ndarray) -> float:
    """
    Max absolute circular correlation at nonzero shifts.
    corr[s] = <a, roll(b, s)> / (||a||*||b||), s in 0..N-1
    Returns max_{s != 0} |corr[s]|; 0 if norms are ~0.
    """
    a64 = a.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)
    na = np.linalg.norm(a64)
    nb = np.linalg.norm(b64)
    if na == 0.0 or nb == 0.0:
        return 0.0
    fa = np.fft.rfft(a64)
    fb = np.fft.rfft(b64)
    # circular cross-correlation via IFFT(FA * conj(FB))
    corr = np.fft.irfft(fa * np.conjugate(fb), n=a64.shape[0]).real / (na * nb)
    if corr.shape[0] > 1:
        return float(np.max(np.abs(corr[1:])))  # exclude shift 0
    return 0.0

def compute_snapshot_divergence_with_penalty(
    S: np.ndarray,
    alpha: float = 0.5,
    lag_low: int = 1,
    lag_high: int = 3,
    beta_shift: float = 0.3,
) -> float:
    """
    S: (M, N) snapshots from t0..T (M >= 2)
    Score = sum_j [ ||S[j]-S[j-q_j]||_2 / sqrt(N) / (1 + alpha*mean(|S[j]|)) ] - beta_shift * sum_j max_corr_nonzero
    where q_j ~ Uniform{lag_low..lag_high}.
    """
    M, N = S.shape
    lags = np.random.randint(lag_low, lag_high + 1, size=M, dtype=np.int32)
    lags[0] = 0

    denomN = float(np.sqrt(N))
    score = 0.0
    shift_pen = 0.0

    for j in range(1, M):
        q = int(lags[j])
        j_prev = max(0, j - q)
        a = S[j].astype(np.float32, copy=False)
        b = S[j_prev].astype(np.float32, copy=False)

        diff = (a - b)
        ss = float(np.sum(diff * diff, dtype=np.float32))
        d = np.sqrt(ss) / denomN

        mean_abs = float(np.mean(np.abs(a), dtype=np.float32))
        score += d / (1.0 + alpha * mean_abs)

        # anti-shift penalty (discourage pure gliders)
        shift_pen += _circular_corr_max_nonzero(a, b)

    return float(score - beta_shift * shift_pen)


# -------------------------
# Simulation + objective
# -------------------------

def simulate_single(
    K: np.ndarray,
    N: int = 64,
    dt: float = 0.05,
    T: int = 320,
    K_snap: int = 20,
    max_abs: float = 5.0,
    alpha: float = 0.5,
    beta_shift: float = 0.3,
) -> Tuple[bool, Optional[float]]:
    """
    Run a single EM with random initial state (no seed), RK4, wrapping boundary.
    Snapshots from t0=T//2 to T every K_snap (including t0).
    Returns: (overflowed, score_if_ok_else_None)
    """
    K = np.asarray(K, dtype=np.float32)
    if K.shape != (3,):
        raise ValueError("K must have shape (3,)")

    a = (0.05 * np.random.standard_normal(N)).astype(np.float32)

    t0 = T // 2
    snaps = []

    for t in range(1, T + 1):
        a = rk4_step(a, dt, K)

        # early stop: non-finite or magnitude overflow
        if (not np.isfinite(a).all()) or np.any(np.abs(a) > max_abs):
            return True, None

        if t == t0 or (t > t0 and ((t - t0) % K_snap) == 0):
            snaps.append(a.copy())

    if len(snaps) < 2:
        return False, 0.0

    S = np.stack(snaps, axis=0)  # (M, N)
    score = compute_snapshot_divergence_with_penalty(
        S, alpha=alpha, lag_low=1, lag_high=3, beta_shift=beta_shift
    )
    return False, float(score)

def objective_loss(K: np.ndarray, **sim_kwargs) -> float:
    """
    CMA-ES minimizes this loss: loss = -score, or huge if overflow/invalid.
    """
    overflow, score = simulate_single(K, **sim_kwargs)
    if overflow or (score is None) or (not np.isfinite(score)):
        return 1e6
    return -float(score)


# -------------------------
# CMA-ES driver
# -------------------------

def run_cma(
    x0: Optional[np.ndarray] = None,
    sigma0: float = 0.3,
    popsize: int = 128,
    bounds: Tuple[float, float] = (-1.0, 1.0),
    maxiter: int = 80,
    sim_kwargs: Optional[dict] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Run CMA-ES to find K maximizing (divergence - beta_shift * anti_shift).
    Returns (best_K, best_score).
    """
    if sim_kwargs is None:
        sim_kwargs = {}

    if x0 is None:
        x0 = np.zeros(3, dtype=np.float32)

    es = cma.CMAEvolutionStrategy(
        x0.tolist(),
        sigma0,
        {
            "bounds": [bounds[0], bounds[1]],
            "popsize": popsize,
            "verb_log": 0,
            "verbose": -9 if not verbose else 1,
        },
    )

    best_loss = np.inf
    best_K = np.array(x0, dtype=np.float32)

    it = 0
    while not es.stop() and it < maxiter:
        it += 1
        cand = es.ask()
        losses = []
        for K in cand:
            K = np.asarray(K, dtype=np.float32)
            loss = objective_loss(K, **sim_kwargs)
            losses.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_K = K.copy()
        es.tell(cand, losses)
        if verbose:
            print(f"[iter {it:03d}] best_loss {best_loss:.6f}  best_K {best_K}")

    best_score = -float(best_loss)
    return best_K, best_score


# -------------------------
# Smoke test + CSV logging
# -------------------------

def smoke_test_and_log(
    csv_path: str = "cma_wrapping_antishift_results.csv",
    N: int = 64,
    dt: float = 0.05,
    T: int = 320,
    K_snap: int = 20,
    max_abs: float = 5.0,
    alpha: float = 0.5,
    beta_shift: float = 0.3,
    x0: Optional[np.ndarray] = None,
    sigma0: float = 0.3,
    popsize: int = 128,
    bounds: Tuple[float, float] = (-1.0, 1.0),
    maxiter: int = 60,
    repeats_eval: int = 5,
) -> Tuple[np.ndarray, float]:
    """
    Run CMA once, re-evaluate the best K a few times, and append a CSV row.
    """
    sim_kwargs = dict(
        N=N, dt=dt, T=T, K_snap=K_snap, max_abs=max_abs, alpha=alpha, beta_shift=beta_shift
    )

    best_K, best_score = run_cma(
        x0=x0, sigma0=sigma0, popsize=popsize, bounds=bounds,
        maxiter=maxiter, sim_kwargs=sim_kwargs, verbose=True
    )

    # Re-evaluate best_K to gauge robustness
    scores = []
    for _ in range(repeats_eval):
        overflow, score = simulate_single(best_K, **sim_kwargs)
        s = -1e6 if overflow or score is None else float(score)
        scores.append(s)
    scores = np.array(scores, dtype=np.float32)
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))

    print("best_K:", best_K, "best_score_once:", best_score)
    print(f"best_K re-eval mean±std over {repeats_eval} runs: {mean_score:.4f} ± {std_score:.4f}")

    header = [
        "timestamp", "k1", "k2", "k3", "score_once", "score_mean", "score_std",
        "N", "dt", "T", "K_snap", "alpha", "beta_shift", "max_abs", "popsize", "sigma0", "maxiter"
    ]
    row = [
        time.strftime("%Y-%m-%d %H:%M:%S"),
        float(best_K[0]), float(best_K[1]), float(best_K[2]),
        float(best_score), mean_score, std_score,
        N, dt, T, K_snap, alpha, beta_shift, max_abs, popsize, sigma0, maxiter
    ]

    try:
        file_exists = False
        try:
            with open(csv_path, "r"):
                file_exists = True
        except FileNotFoundError:
            file_exists = False
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(header)
            w.writerow(row)
        print(f"Appended results to {csv_path}")
    except Exception as e:
        print("CSV write failed:", e)

    return best_K, best_score


if __name__ == "__main__":
    # Minimal smoke run — tune popsize/maxiter on your box as needed
    best_K, best_score = smoke_test_and_log(
        csv_path="cma_wrapping_antishift_results.csv",
        N=64, dt=0.05, T=320, K_snap=20, max_abs=5.0,
        alpha=0.5,           # normalization strength
        beta_shift=0.3,      # anti-shift penalty weight
        x0=None, sigma0=0.3, popsize=128, bounds=(-1.0, 1.0), maxiter=60,
        repeats_eval=5,
    )
    print("Done. Best:", best_K, "score:", best_score)