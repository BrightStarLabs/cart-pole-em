
import numpy as np

def neighbors_zero_pad(a):
    """Return (L, C, R) with zero padding for a 1D state a of shape (N,)."""
    N = a.shape[0]
    L = np.empty_like(a)
    R = np.empty_like(a)
    L[0] = 0.0
    L[1:] = a[:-1]
    R[-1] = 0.0
    R[:-1] = a[1:]
    return L, a, R

def rhs(a, K):
    """Compute da/dt for EM with K=(k1..k5). a shape (N,)."""
    k1, k2, k3, k4, k5 = K
    L, C, R = neighbors_zero_pad(a)
    return k1*L + k2*C + k3*R + k4*(C*R) + k5*(C*L)

def rk4_step(a, dt, K):
    """One RK4 step for a' = f(a)."""
    s1 = rhs(a, K)
    s2 = rhs(a + 0.5*dt*s1, K)
    s3 = rhs(a + 0.5*dt*s2, K)
    s4 = rhs(a + dt*s3, K)
    return a + (dt/6.0)*(s1 + 2*s2 + 2*s3 + s4)

def simulate_single_em(
    K,
    N=64,
    dt=0.05,
    T=256,
    max_abs_a=5.0,
    seed=0,
    A0=None,
    return_traj=False,
):
    """
    CPU NumPy simulator for a SINGLE EM with zero padding + RK4.

    Args:
        K: array-like of shape (5,), EM parameters [k1..k5].
        N: number of cells.
        dt: integrator step.
        T: number of steps.
        max_abs_a: overflow threshold. If exceeded, state is zeroed and frozen.
        seed: RNG seed if A0 is None.
        A0: optional initial state (N,). If None, random normal 0.05 std.
        return_traj: if True, returns a trajectory (T+1, N). Otherwise returns final state only.

    Returns:
        (a_final, overflow, traj_or_none)
    """
    K = np.asarray(K, dtype=np.float32)
    if A0 is None:
        rng = np.random.default_rng(seed)
        a = (0.1 * rng.standard_normal(N)).astype(np.float32)
    else:
        a = np.asarray(A0, dtype=np.float32).copy()

    if return_traj:
        traj = np.empty((T+1, N), dtype=np.float32)
        traj[0] = a
    else:
        traj = None

    overflow = False
    for t in range(1, T+1):
        a = rk4_step(a, dt, K)
        if np.any(np.abs(a) > max_abs_a):
            overflow = True
            a[:] = 0.0
            if return_traj:
                traj[t] = a
            break
        if return_traj:
            traj[t] = a

    return a, overflow, traj

if __name__ == "__main__":
    # Small self-test
    K = np.array([0.30, -0.80, 0.30, 0.05, 0.05], dtype=np.float32)
    a_final, overflow, traj = simulate_single_em(K, N=64, dt=0.05, T=128, return_traj=True)
    print("Final state shape:", a_final.shape, "Overflow:", overflow, "Traj:", None if traj is None else traj.shape)
