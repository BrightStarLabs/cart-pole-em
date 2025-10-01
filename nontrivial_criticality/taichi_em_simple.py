
import taichi as ti
import numpy as np

ti.init(arch=ti.cuda, default_fp=ti.f32)

@ti.data_oriented
class SimpleTaichiEMS:
    def __init__(self, B, N, dt, T, max_abs_a=5.0, seed=0):
        self.B, self.N = int(B), int(N)
        self.dt, self.T = float(dt), int(T)
        self.max_abs_a = float(max_abs_a)

        rng = np.random.default_rng(seed)
        A0 = (0.05 * rng.standard_normal((B, N))).astype(np.float32)
        K0 = rng.uniform(low=-0.5, high=0.5, size=(B, 5)).astype(np.float32)

        self.A   = ti.field(dtype=ti.f32, shape=(B, N))
        self.tmp = ti.field(dtype=ti.f32, shape=(B, N))
        self.k1f = ti.field(dtype=ti.f32, shape=(B, N))
        self.k2f = ti.field(dtype=ti.f32, shape=(B, N))
        self.k3f = ti.field(dtype=ti.f32, shape=(B, N))
        self.k4f = ti.field(dtype=ti.f32, shape=(B, N))

        self.K = ti.field(dtype=ti.f32, shape=(B, 5))   # per-EM params [k1..k5]
        self.overflow = ti.field(dtype=ti.i32, shape=(B,))
        self.exceeded = ti.field(dtype=ti.i32, shape=(B,))

        self.A.from_numpy(A0)
        self.K.from_numpy(K0)
        self.clear_flags()

    @ti.kernel
    def clear_flags(self):
        for b in range(self.B):
            self.overflow[b] = 0
            self.exceeded[b] = 0

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

    @ti.kernel
    def rhs_from_A(self, out_stage: ti.i32):
        for b, i in self.A:
            if self.overflow[b] == 1:
                if out_stage == 1: self.k1f[b, i] = 0.0
                elif out_stage == 2: self.k2f[b, i] = 0.0
                elif out_stage == 3: self.k3f[b, i] = 0.0
                else: self.k4f[b, i] = 0.0
                continue
            k1c = self.K[b, 0]; k2c = self.K[b, 1]; k3c = self.K[b, 2]; k4c = self.K[b, 3]; k5c = self.K[b, 4]
            C = self.A[b, i]
            L = self._L_A(b, i)
            R = self._R_A(b, i)
            v = k1c*L + k2c*C + k3c*R + k4c*(C*R) + k5c*(C*L)
            if out_stage == 1: self.k1f[b, i] = v
            elif out_stage == 2: self.k2f[b, i] = v
            elif out_stage == 3: self.k3f[b, i] = v
            else: self.k4f[b, i] = v

    @ti.kernel
    def rhs_from_tmp(self, out_stage: ti.i32):
        for b, i in self.A:
            if self.overflow[b] == 1:
                if out_stage == 2: self.k2f[b, i] = 0.0
                elif out_stage == 3: self.k3f[b, i] = 0.0
                else: self.k4f[b, i] = 0.0
                continue
            k1c = self.K[b, 0]; k2c = self.K[b, 1]; k3c = self.K[b, 2]; k4c = self.K[b, 3]; k5c = self.K[b, 4]
            C = self.tmp[b, i]
            L = self._L_tmp(b, i)
            R = self._R_tmp(b, i)
            v = k1c*L + k2c*C + k3c*R + k4c*(C*R) + k5c*(C*L)
            if out_stage == 2: self.k2f[b, i] = v
            elif out_stage == 3: self.k3f[b, i] = v
            else: self.k4f[b, i] = v

    @ti.kernel
    def combine(self, stage: ti.i32, dt: ti.f32):
        for b, i in self.A:
            if self.overflow[b] == 1:
                self.tmp[b, i] = 0.0
                continue
            if stage == 1:
                self.tmp[b, i] = self.A[b, i] + 0.5*dt*self.k1f[b, i]
            elif stage == 2:
                self.tmp[b, i] = self.A[b, i] + 0.5*dt*self.k2f[b, i]
            else:
                self.tmp[b, i] = self.A[b, i] + dt*self.k3f[b, i]

    @ti.kernel
    def final_update(self, dt: ti.f32, max_abs: ti.f32):
        for b, i in self.A:
            if self.overflow[b] == 1:
                self.A[b, i] = 0.0
                continue
            self.A[b, i] = self.A[b, i] + (dt/6.0)*(
                self.k1f[b, i] + 2.0*self.k2f[b, i] + 2.0*self.k3f[b, i] + self.k4f[b, i]
            )
            if ti.abs(self.A[b, i]) > max_abs:
                self.exceeded[b] = 1

    @ti.kernel
    def apply_overflow(self):
        for b in range(self.B):
            if self.exceeded[b] == 1 and self.overflow[b] == 0:
                self.overflow[b] = 1
                for i in range(self.N):
                    self.A[b, i] = 0.0
            self.exceeded[b] = 0

    def step(self):
        dt = self.dt
        self.rhs_from_A(1)         # k1f = f(A)
        self.combine(1, dt)        # tmp = A + 0.5*dt*k1
        self.rhs_from_tmp(2)       # k2f = f(tmp)
        self.combine(2, dt)        # tmp = A + 0.5*dt*k2
        self.rhs_from_tmp(3)       # k3f = f(tmp)
        self.combine(3, dt)        # tmp = A + dt*k3
        self.rhs_from_tmp(4)       # k4f = f(tmp)
        self.final_update(dt, self.max_abs_a)
        self.apply_overflow()

    def run(self):
        for _ in range(self.T):
            self.step()

    def state_numpy(self):
        return self.A.to_numpy()

    def overflow_numpy(self):
        return self.overflow.to_numpy()

    def K_numpy(self):
        return self.K.to_numpy()


if __name__ == "__main__":
    B, N = 2048, 64
    dt, T = 0.05, 200
    sim = SimpleTaichiEMS(B=B, N=N, dt=dt, T=T, max_abs_a=5.0, seed=0)
    sim.run()
    A = sim.state_numpy()
    ov = sim.overflow_numpy()
    print("State:", A.shape, "Overflowed:", int(ov.sum()), "/", B)
