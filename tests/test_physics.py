"""Analytic sanity checks for the physics core. Run: python -m tests.test_physics"""
import numpy as np
from crashlab import vehicles as V
from crashlab.dynamics import new_state, vehicle_step, forward_speed
from crashlab.collision import apply_impulse, kinetic_energy

G = 9.81
results = {}


def skid_distance(spec, v0, mu, dt=0.002):
    st = new_state(1, 0, 0, 0, v0)
    while forward_speed(st)[0] > 0.02:
        vehicle_step(spec, st, 0.0, -1.5 * G, mu, dt)
        if st["x"][0] > 500: break
    return st["x"][0]


def t_skid():
    out = []
    for name, spec in [("sedan", V.sedan()), ("suv", V.suv()), ("truck", V.truck()),
                       ("motorcycle", V.motorcycle())]:
        d = skid_distance(spec, 20.0, 0.8)
        ref = 20.0 ** 2 / (2 * 0.8 * G)
        out.append((name, round(d, 2), round(ref, 2), round(100 * (d - ref) / ref, 2)))
    return out


def t_momentum():
    # oblique impact, no external forces during the impulse: momentum must be conserved
    sa, sb = V.sedan(), V.suv()
    A = new_state(1, 0, 0, 0, 15.0)
    B = new_state(1, 4.0, 0.5, np.pi, 10.0)
    n = np.array([1.0, 0.0]); P = np.array([2.0, 0.25])
    p0 = np.array([sa.mass * A["vx"][0] + sb.mass * B["vx"][0], sa.mass * A["vy"][0] + sb.mass * B["vy"][0]])
    # angular momentum about the contact point
    def L(sp, s):
        rx, ry = s["x"][0] - P[0], s["y"][0] - P[1]
        return sp.izz * s["r"][0] + sp.mass * (rx * s["vy"][0] - ry * s["vx"][0])
    L0 = L(sa, A) + L(sb, B); ke0 = kinetic_energy(sa, A)[0] + kinetic_energy(sb, B)[0]
    apply_impulse(sa, A, sb, B, P, n, 0.3, 0.2)
    p1 = np.array([sa.mass * A["vx"][0] + sb.mass * B["vx"][0], sa.mass * A["vy"][0] + sb.mass * B["vy"][0]])
    L1 = L(sa, A) + L(sb, B); ke1 = kinetic_energy(sa, A)[0] + kinetic_energy(sb, B)[0]
    return dict(dp=float(np.abs(p1 - p0).max()), dL=float(abs(L1 - L0)), ke_loss_kj=round((ke0 - ke1) / 1000, 2))


def t_plastic():
    # central head-on, e=0, no friction: energy loss = 1/2 * mu_red * vc^2 ; common final velocity
    sa, sb = V.sedan(), V.hatchback()
    A = new_state(1, 0, 0, 0, 15.0); B = new_state(1, 4.2, 0, np.pi, 10.0)
    ke0 = kinetic_energy(sa, A)[0] + kinetic_energy(sb, B)[0]
    apply_impulse(sa, A, sb, B, np.array([2.1, 0.0]), np.array([1.0, 0.0]), 0.0, 0.0)
    mred = sa.mass * sb.mass / (sa.mass + sb.mass)
    ke1 = kinetic_energy(sa, A)[0] + kinetic_energy(sb, B)[0]
    return dict(dv_common=float(abs(A["vx"][0] - B["vx"][0])), loss_sim=round((ke0 - ke1) / 1000, 3),
                loss_theory=round(0.5 * mred * 25.0 ** 2 / 1000, 3))


def t_elastic():
    sa = sb = V.sedan()
    A = new_state(1, 0, 0, 0, 12.0); B = new_state(1, 4.5, 0, np.pi, 6.0)
    apply_impulse(sa, A, sb, B, np.array([2.25, 0.0]), np.array([1.0, 0.0]), 1.0, 0.0)
    return dict(A_after=float(A["vx"][0]), B_after=float(B["vx"][0]))


def t_dt_convergence():
    spec = V.sedan()
    r = []
    for dt in (0.004, 0.002, 0.001):
        st = new_state(1, 0, 0, 0.0, 22.0)
        st["delta"][:] = 0.0
        for k in range(int(3.0 / dt)):
            steer = 0.06 if k * dt < 1.0 else -0.03
            vehicle_step(spec, st, steer, -0.4 * G if k * dt > 0.5 else 0.0, 0.8, dt, True)
        r.append((dt, round(float(st["x"][0]), 3), round(float(st["y"][0]), 3), round(float(st["psi"][0]), 4)))
    return r


if __name__ == "__main__":
    print("skid-to-speed (v0=20 m/s, mu=0.8, locked): name, sim m, v^2/2mu g m, error %")
    for r in t_skid(): print("  ", r)
    print("momentum:", t_momentum())
    print("plastic :", t_plastic())
    print("elastic :", t_elastic())
    print("dt conv :", t_dt_convergence())
