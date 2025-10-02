
import os
import csv
import taichi as ti
import numpy as np

ti.init(arch=ti.cuda, default_fp=ti.f32)

@ti.data_oriented
class EMSnapshotDivergence:
    """
    Batched 1-D EM with RK4 (zero padding).
    Snapshots are recorded only from t0 = T//2 to T, every K_snap steps, including t0.
    After the run, compute per-EM snapshot-divergence scores with random lag q in {1,2,3}:
        score[b] = sum_j || S[j,b,:] - S[j-q_j,b,:] ||_2 / sqrt(N)
    Overflow during the run (|A|>max_abs or NaN) -> flag + zero + freeze.
    No additional filters are applied in scoring.
    """

    def __init__(self, B, N, dt, T, K_snap, max_abs_a=5.0, seed=0, K_init=None, A0=None):
        self.B, self.N = int(B), int(N)
        self.dt, self.T = float(dt), int(T)
        self.K_snap = int(K_snap)
        assert self.K_snap > 0
        self.max_abs_a = float(max_abs_a)

        # Snapshot window start t0 and count
        self.t0 = (2*self.T) // 3
        remain = self.T - self.t0
        self.M = remain // self.K_snap
        self.num_snaps = self.M + 1  # indices 0..M (t0 included)

        rng = np.random.default_rng(seed)
        if A0 is None:
            A0 = np.clip((0.5 * rng.standard_normal((B, N))).astype(np.float32), -1.0, 1.0)
        else:
            A0 = np.asarray(A0, dtype=np.float32)
            assert A0.shape == (B, N)

        if K_init is None:
            K0 = rng.uniform(low=-0.8, high=0.8, size=(B, 5)).astype(np.float32)
        else:
            K0 = np.asarray(K_init, dtype=np.float32)
            assert K0.shape == (B, 5)

        # Fields
        self.A      = ti.field(dtype=ti.f32, shape=(B, N))
        self.A_prev = ti.field(dtype=ti.f32, shape=(B, N))
        self.tmp    = ti.field(dtype=ti.f32, shape=(B, N))
        self.s1     = ti.field(dtype=ti.f32, shape=(B, N))
        self.s2     = ti.field(dtype=ti.f32, shape=(B, N))
        self.s3     = ti.field(dtype=ti.f32, shape=(B, N))
        self.s4     = ti.field(dtype=ti.f32, shape=(B, N))

        self.K = ti.field(dtype=ti.f32, shape=(B, 5))
        self.overflow = ti.field(dtype=ti.i32, shape=(B,))
        self.exceeded = ti.field(dtype=ti.i32, shape=(B,))

        # Snapshots restricted to t in [t0, T]
        self.snaps = ti.field(dtype=ti.f32, shape=(self.num_snaps, B, N))

        # Init
        self.A.from_numpy(A0)
        self.A_prev.from_numpy(A0)
        self.K.from_numpy(K0)
        self._clear_flags()

    @ti.kernel
    def _clear_flags(self):
        for b in range(self.B):
            self.overflow[b] = 0
            self.exceeded[b] = 0

    # ---------- neighbors (zero-padding) ----------
    @ti.func
    def _L_A(self, b, i):
        return self.A[b, i-1] if i > 0 else 0.0

    @ti.func
    def _R_A(self, b, i):
        return self.A[b, i+1] if i < self.N - 1 else 0.0

    @ti.func
    def _L_tmp(self, b, i):
        return self.tmp[b, i-1] if i > 0 else 0.0

    @ti.func
    def _R_tmp(self, b, i):
        return self.tmp[b, i+1] if i < self.N - 1 else 0.0

    # ---------- RHS ----------
    @ti.kernel
    def rhs_from_A(self):
        for b, i in self.A:
            if self.overflow[b] == 1:
                self.s1[b, i] = 0.0
                continue
            k1c = self.K[b, 0]; k2c = self.K[b, 1]; k3c = self.K[b, 2]; k4c = self.K[b, 3]; k5c = self.K[b, 4]
            C = self.A[b, i]
            L = self._L_A(b, i)
            R = self._R_A(b, i)
            self.s1[b, i] = k1c*L + k2c*C + k3c*R #-0.1* C**3 #+ k4c*(C*R) + k5c*(C*L)

    @ti.kernel
    def rhs_from_tmp(self, stage: ti.i32):
        for b, i in self.A:
            if self.overflow[b] == 1:
                if stage == 2: self.s2[b, i] = 0.0
                elif stage == 3: self.s3[b, i] = 0.0
                else: self.s4[b, i] = 0.0
                continue
            k1c = self.K[b, 0]; k2c = self.K[b, 1]; k3c = self.K[b, 2]; k4c = self.K[b, 3]; k5c = self.K[b, 4]
            C = self.tmp[b, i]
            L = self._L_tmp(b, i)
            R = self._R_tmp(b, i)
            v = k1c*L + k2c*C + k3c*R #-0.1* C**3 #+ k4c*(C*R) + k5c*(C*L)
            if stage == 2: self.s2[b, i] = v
            elif stage == 3: self.s3[b, i] = v
            else: self.s4[b, i] = v

    # ---------- RK4 combines ----------
    @ti.kernel
    def combine_half_k1(self, dt: ti.f32):
        for b, i in self.A:
            if self.overflow[b] == 1:
                self.tmp[b, i] = 0.0
            else:
                self.tmp[b, i] = self.A[b, i] + 0.5 * dt * self.s1[b, i]

    @ti.kernel
    def combine_half_k2(self, dt: ti.f32):
        for b, i in self.A:
            if self.overflow[b] == 1:
                self.tmp[b, i] = 0.0
            else:
                self.tmp[b, i] = self.A[b, i] + 0.5 * dt * self.s2[b, i]

    @ti.kernel
    def combine_full_k3(self, dt: ti.f32):
        for b, i in self.A:
            if self.overflow[b] == 1:
                self.tmp[b, i] = 0.0
            else:
                self.tmp[b, i] = self.A[b, i] + dt * self.s3[b, i]

    @ti.kernel
    def final_update_and_check(self, dt: ti.f32, max_abs: ti.f32):
        for b, i in self.A:
            if self.overflow[b] == 1:
                self.A[b, i] = 0.0
            else:
                self.A_prev[b, i] = self.A[b, i]
                self.A[b, i] = self.A[b, i] + (dt/6.0) * (
                    self.s1[b, i] + 2.0*self.s2[b, i] + 2.0*self.s3[b, i] + self.s4[b, i]
                )
                x = self.A[b, i]
                if (x != x) or (ti.abs(x) > max_abs):
                    self.exceeded[b] = 1

    @ti.kernel
    def apply_overflow(self):
        for b in range(self.B):
            if self.exceeded[b] == 1 and self.overflow[b] == 0:
                self.overflow[b] = 1
                for i in range(self.N):
                    self.A[b, i] = 0.0
            self.exceeded[b] = 0

    @ti.kernel
    def write_snapshot(self, snap_idx: ti.i32):
        for b, i in self.A:
            self.snaps[snap_idx, b, i] = self.A[b, i]

    def step(self, t):
        dt = self.dt
        self.rhs_from_A()
        self.combine_half_k1(dt)
        self.rhs_from_tmp(2)
        self.combine_half_k2(dt)
        self.rhs_from_tmp(3)
        self.combine_full_k3(dt)
        self.rhs_from_tmp(4)
        self.final_update_and_check(dt, self.max_abs_a)
        self.apply_overflow()

    def run(self):
        # record snapshots only from t0 onward
        snap_idx = 0
        for t in range(1, self.T + 1):
            self.step(t)
            if t == self.t0 or (t > self.t0 and (t - self.t0) % self.K_snap == 0):
                if snap_idx < self.num_snaps:
                    self.write_snapshot(snap_idx)
                    snap_idx += 1

    def snapshots_numpy(self):
        return self.snaps.to_numpy()

    def overflow_numpy(self):
        return self.overflow.to_numpy()

    def K_numpy(self):
        return self.K.to_numpy()

    @staticmethod
    def compute_snapshot_divergence(snaps, lag_choices=(1,2,3), seed=0, return_sorted_idx=True):
        S, B, N = snaps.shape
        rng = np.random.default_rng(seed)
        lags = rng.integers(low=min(lag_choices), high=max(lag_choices)+1, size=S)
        lags[0] = 0
        scores = np.zeros((B,), dtype=np.float64)
        denom = float(np.sqrt(N))
        for j in range(1, S):
            q = int(lags[j])
            j_prev = max(0, j - q)
            diff = snaps[j] - snaps[j_prev]
            ss = np.sum(diff.astype(np.float64)**2, axis=1, dtype=np.float64)
            scores += np.sqrt(ss) / denom
        scores = scores.astype(np.float32)
        if return_sorted_idx:
            idx = np.argsort(-scores)
            return scores, idx
        return scores

# ---------------------
# Smoke test that appends to CSV across seeds
# ---------------------
if __name__ == "__main__":
    out_csv = "em_snapshot_divergence_results.csv"
    write_header = not os.path.exists(out_csv)

    seeds = [0, 1, 2, 3, 4]
    B, N = 4096, 64
    dt, T, K_snap = 0.05, 320, 20
    top_k = 100

    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["seed", "em_idx", "score", "k1", "k2", "k3", "k4", "k5"])

        for seed in seeds:
            sim = EMSnapshotDivergence(B=B, N=N, dt=dt, T=T, K_snap=K_snap, max_abs_a=5.0, seed=seed)
            sim.run()
            snaps = sim.snapshots_numpy()
            K = sim.K_numpy()
            scores, idx_sorted = EMSnapshotDivergence.compute_snapshot_divergence(
                snaps, lag_choices=(1,2,3), seed=seed, return_sorted_idx=True
            )
            sel = idx_sorted[:top_k]
            for em_idx in sel:
                kvec = K[em_idx]
                w.writerow([seed, int(em_idx), float(scores[em_idx]),
                            float(kvec[0]), float(kvec[1]), float(kvec[2]), float(kvec[3]), float(kvec[4])])

    print(f"Appended results to {out_csv}")
