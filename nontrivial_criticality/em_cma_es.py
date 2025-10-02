
import time
import csv
from typing import Tuple, Optional

import numpy as np

try:
    import cma
except Exception as e:
    raise ImportError("The 'cma' package is required. Install with: pip install cma") from e


# -------------------------
# EM dynamics (3 params)
# -------------------------

def neighbors_zero_pad(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = a.shape[0]
    L = np.empty_like(a)
    R = np.empty_like(a)
    L[0] = 0.0
    L[1:] = a[:-1]
    R[-1] = 0.0
    R[:-1] = a[1:]
    return L, a, R


def rhs_linear(a: np.ndarray, K: np.ndarray) -> np.ndarray:
    k1, k2, k3 = K
    L, C, R = neighbors_zero_pad(a)
    return k1 * L + k2 * C + k3 * R


def rk4_step(a: np.ndarray, dt: float, K: np.ndarray) -> np.ndarray:
    s1 = rhs_linear(a, K)
    s2 = rhs_linear(a + 0.5 * dt * s1, K)
    s3 = rhs_linear(a + 0.5 * dt * s2, K)
    s4 = rhs_linear(a + dt * s3, K)
    return a + (dt / 6.0) * (s1 + 2 * s2 + 2 * s3 + s4)


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
) -> Tuple[bool, Optional[float]]:
    """
    Run a single EM with random initial state (no seed), RK4, zero padding.
    Snapshots from t0=T//2 to T every K_snap (including t0).
    Score = sum_j ||S[j]-S[j-q_j]||_2 / sqrt(N) / (1 + alpha*mean(|S[j]|)),
    with q_j ~ Uniform{1,2,3}.

    Returns: (overflowed, score_if_ok_else_None)
    """
    K = np.asarray(K, dtype=np.float32)
    if K.shape != (3,):
        raise ValueError("K must have shape (3,)")

    a = np.clip((0.4 * np.random.standard_normal(N)).astype(np.float32),-1,1)

    t0 = T // 2
    snaps = []

    for t in range(1, T + 1):
        a = rk4_step(a, dt, K)

        if (not np.isfinite(a).all()) or np.any(np.abs(a) > max_abs):
            return True, None

        if t == t0 or (t > t0 and ((t - t0) % K_snap) == 0):
            snaps.append(a.copy())

    if len(snaps) < 2:
        return False, 0.0

    S = np.stack(snaps, axis=0)  # (M, N)
    M = S.shape[0]
    denom_N = float(np.sqrt(N))

    lags = np.random.randint(1, 7, size=M, dtype=np.int32)  # 1..3
    lags[0] = 0

    score = 0.0
    for j in range(1, M):
        q = int(lags[j])
        j_prev = max(0, j - q)
        diff = (S[j] - S[j_prev]).astype(np.float32)
        ss = float(np.sum(diff * diff, dtype=np.float32))
        d = np.sqrt(ss) / denom_N
        mean_abs = float(np.mean(np.abs(S[j]), dtype=np.float32))
        score += d / (1.0 + alpha * mean_abs)

    return False, float(score)


def objective_loss(
    K: np.ndarray,
    **sim_kwargs,
) -> float:
    overflow, score = simulate_single(K, **sim_kwargs)
    if overflow or (score is None) or (not np.isfinite(score)):
        return 1e6
    return -float(score)


# -------------------------
# CMA-ES driver
# -------------------------

def run_cma(
    x0: np.ndarray = np.zeros(3, dtype=np.float32),
    sigma0: float = 0.3,
    popsize: int = 64,
    bounds: Tuple[float, float] = (-1.0, 1.0),
    maxiter: int = 80,
    sim_kwargs: Optional[dict] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    if sim_kwargs is None:
        sim_kwargs = {}

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
            print(f"[iter {it:03d}] best_loss: {best_loss:.6f} best_K: {best_K}")

    best_score = -float(best_loss)
    return best_K, best_score


# -------------------------
# Smoke test runner
# -------------------------

def smoke_test_and_log(
    csv_path: str = "cma_em_best.csv",
    N: int = 64,
    dt: float = 0.05,
    T: int = 320,
    K_snap: int = 20,
    max_abs: float = 5.0,
    alpha: float = 0.5,
    x0: np.ndarray = np.zeros(3, dtype=np.float32),
    sigma0: float = 0.3,
    popsize: int = 64,
    bounds: Tuple[float, float] = (-1.0, 1.0),
    maxiter: int = 50,
    repeats_eval: int = 5,
) -> Tuple[np.ndarray, float]:
    sim_kwargs = dict(N=N, dt=dt, T=T, K_snap=K_snap, max_abs=max_abs, alpha=alpha)

    best_K, best_score = run_cma(
        x0=x0, sigma0=sigma0, popsize=popsize, bounds=bounds,
        maxiter=maxiter, sim_kwargs=sim_kwargs, verbose=True
    )

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
        "N", "dt", "T", "K_snap", "alpha", "max_abs", "popsize", "sigma0", "maxiter"
    ]
    row = [
        time.strftime("%Y-%m-%d %H:%M:%S"),
        float(best_K[0]), float(best_K[1]), float(best_K[2]),
        float(best_score), mean_score, std_score,
        N, dt, T, K_snap, alpha, max_abs, popsize, sigma0, maxiter
    ]

    from pathlib import Path
    file_exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        w.writerow(row)

    print(f"Appended results to {csv_path}")
    return best_K, best_score


if __name__ == "__main__":
    best_K, best_score = smoke_test_and_log(
        csv_path="cma_em_best.csv",
        N=64, dt=0.05, T=320, K_snap=20, max_abs=5.0, alpha=0.5,
        x0=np.zeros(3, dtype=np.float32), sigma0=0.3, popsize=64, bounds=(-1.0, 1.0), maxiter=40,
        repeats_eval=5,
    )
    print("Done. Best:", best_K, "score:", best_score)
