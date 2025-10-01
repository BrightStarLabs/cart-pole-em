
import taichi as ti
import numpy as np

ti.init(arch=ti.cuda, default_fp=ti.f32)

@ti.data_oriented
class EMDiscovery:
    def __init__(self, B, N, dt, T, max_abs_a=5.0, beta_ema=0.1, seed=0):
        self.B, self.N = int(B), int(N)
        self.dt, self.T = float(dt), int(T)
        self.max_abs_a = float(max_abs_a)
        self.beta_ema = float(beta_ema)
        self.half_start = self.T // 2

        rng = np.random.default_rng(seed)
        A0 = (0.05 * rng.standard_normal((B, N))).astype(np.float32)
        K0 = rng.uniform(low=-0.5, high=0.5, size=(B, 5)).astype(np.float32)

        # States
        self.A        = ti.field(dtype=ti.f32, shape=(B, N))
        self.A_prev   = ti.field(dtype=ti.f32, shape=(B, N))
        self.tmp      = ti.field(dtype=ti.f32, shape=(B, N))
        # RK4 slopes
        self.s1 = ti.field(dtype=ti.f32, shape=(B, N))
        self.s2 = ti.field(dtype=ti.f32, shape=(B, N))
        self.s3 = ti.field(dtype=ti.f32, shape=(B, N))
        self.s4 = ti.field(dtype=ti.f32, shape=(B, N))
        # Per-EM params
        self.K = ti.field(dtype=ti.f32, shape=(B, 5))  # k1..k5 per EM
        # Overflow & step counters
        self.overflow = ti.field(dtype=ti.i32, shape=(B,))
        self.exceeded = ti.field(dtype=ti.i32, shape=(B,))  # scratch per step
        self.steps_done = ti.field(dtype=ti.i32, shape=(B,))
        # Per-step per-cell diffs (buffer for reductions, no atomics)
        self.delta_abs = ti.field(dtype=ti.f32, shape=(B, N))
        self.sign_prev = ti.field(dtype=ti.i8,  shape=(B, N))
        self.sign_cur  = ti.field(dtype=ti.i8,  shape=(B, N))

        # Per-EM accumulators
        self.activity_sum   = ti.field(dtype=ti.f32, shape=(B,))  # last-half sum of mean |ΔA|
        self.ema_change     = ti.field(dtype=ti.f32, shape=(B,))  # EMA of mean |ΔA|
        self.sign_flips_sum = ti.field(dtype=ti.f32, shape=(B,))  # last-half sum of flip ratios

        self.A.from_numpy(A0)
        self.A_prev.from_numpy(A0)
        self.K.from_numpy(K0)

        self._init_flags_and_metrics()

    @ti.kernel
    def _init_flags_and_metrics(self):
        for b in range(self.B):
            self.overflow[b] = 0
            self.exceeded[b] = 0
            self.steps_done[b] = 0
            self.activity_sum[b] = 0.0
            self.ema_change[b] = 0.0
            self.sign_flips_sum[b] = 0.0
        for b, i in self.A:
            self.sign_prev[b, i] = ti.i8(1 if self.A[b, i] >= 0.0 else -1)
            self.sign_cur[b, i]  = self.sign_prev[b, i]
            self.delta_abs[b, i] = 0.0

    # ---------- neighbor helpers (zero padding) ----------
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
            self.s1[b, i] = k1c*L + k2c*C + k3c*R + k4c*(C*R) + k5c*(C*L)

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
            v = k1c*L + k2c*C + k3c*R + k4c*(C*R) + k5c*(C*L)
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
    def final_update(self, dt: ti.f32, max_abs: ti.f32):
        for b, i in self.A:
            if self.overflow[b] == 1:
                self.A[b, i] = 0.0
            else:
                self.A_prev[b, i] = self.A[b, i]  # keep previous
                self.A[b, i] = self.A[b, i] + (dt/6.0) * (
                    self.s1[b, i] + 2.0*self.s2[b, i] + 2.0*self.s3[b, i] + self.s4[b, i]
                )
                if ti.abs(self.A[b, i]) > max_abs:
                    self.exceeded[b] = 1

    # ---------- per-step measurement without atomics ----------
    @ti.kernel
    def compute_delta_and_sign(self):
        for b, i in self.A:
            if self.overflow[b] == 1:
                self.delta_abs[b, i] = 0.0
                self.sign_cur[b, i] = 0
            else:
                da = self.A[b, i] - self.A_prev[b, i]
                self.delta_abs[b, i] = ti.abs(da)
                self.sign_cur[b, i] = ti.i8(1 if self.A[b, i] >= 0.0 else -1)

    @ti.kernel
    def reduce_per_em(self, start_accum_step: ti.i32, cur_step: ti.i32):
        for b in range(self.B):
            if self.overflow[b] == 1:
                continue
            # reduce over N locally (no atomics)
            s = 0.0
            flips = 0
            for i in range(self.N):
                s += self.delta_abs[b, i]
                if self.sign_cur[b, i] != self.sign_prev[b, i]:
                    flips += 1
                # update sign_prev for next step
                self.sign_prev[b, i] = self.sign_cur[b, i]
            mean_da = s / float(self.N)

            # EMA
            self.ema_change[b] = (1.0 - self.beta_ema) * self.ema_change[b] + self.beta_ema * mean_da

            # last-half accumulations
            if cur_step >= start_accum_step:
                self.activity_sum[b] += mean_da
                self.sign_flips_sum[b] += float(flips) / float(self.N)

            self.steps_done[b] += 1

    @ti.kernel
    def apply_overflow(self):
        for b in range(self.B):
            if self.exceeded[b] == 1 and self.overflow[b] == 0:
                self.overflow[b] = 1
                for i in range(self.N):
                    self.A[b, i] = 0.0
            self.exceeded[b] = 0  # reset for next step

    def step(self, t_idx: int):
        dt = self.dt
        self.rhs_from_A()
        self.combine_half_k1(dt)
        self.rhs_from_tmp(2)
        self.combine_half_k2(dt)
        self.rhs_from_tmp(3)
        self.combine_full_k3(dt)
        self.rhs_from_tmp(4)
        self.final_update(dt, self.max_abs_a)
        self.apply_overflow()
        self.compute_delta_and_sign()
        self.reduce_per_em(self.half_start, t_idx)

    def run(self):
        for t in range(self.T):
            self.step(t)

    # ---------- retrieval ----------
    def state_numpy(self):
        return self.A.to_numpy()

    def K_numpy(self):
        return self.K.to_numpy()

    def metrics_numpy(self):
        return (
            self.activity_sum.to_numpy(),
            self.ema_change.to_numpy(),
            self.sign_flips_sum.to_numpy(),
            self.overflow.to_numpy(),
            self.steps_done.to_numpy(),
        )

    # ---------- selection (Python side) ----------
    @staticmethod
    def select_interesting(K, activity, ema, flips, overflow, *,
                           eps_converge=2e-4, lambda_flip=0.1, top_k=1000):
        ok = (overflow == 0) & (ema >= eps_converge)
        score = activity + lambda_flip * flips
        score[~ok] = -np.inf
        idx = np.argpartition(-score, kth=min(top_k, len(score)-1))[:top_k]
        idx = idx[np.argsort(-score[idx])]
        return idx, score
