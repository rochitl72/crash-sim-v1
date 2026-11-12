"""Planar 3-DOF rigid-body vehicle dynamics with per-wheel tire forces.

Everything is vectorised over a batch dimension N so the *same* code drives both
the single-scene animation (N=1) and the Monte-Carlo inverse solver (N=thousands).

Tire model (per wheel):
  * normal load Fz = static share +- longitudinal / lateral load transfer
  * rolling wheel : friction-ellipse limited force, lateral force from slip angle
                    Fy = -mu*Fz*tanh(c_alpha*alpha/mu)
  * locked wheel  : sliding Coulomb force opposing the contact-patch velocity
  * toppled body  : (motorcycles after impact) isotropic Coulomb sliding of the body
"""
import numpy as np
from .vehicles import G

RHO = 1.2
EPS_ALPHA = 1.0     # m/s  low-speed regularisation of slip angle
EPS_SLIDE = 0.25    # m/s  low-speed regularisation of sliding friction
STEER_RATE = 1.0    # rad/s


def new_state(n, x=0.0, y=0.0, psi=0.0, speed=0.0):
    f = lambda v: np.array(np.broadcast_to(np.asarray(v, float), (n,)))
    p = f(psi)
    s = f(speed)
    return dict(x=f(x), y=f(y), psi=p, vx=s * np.cos(p), vy=s * np.sin(p), r=f(0.0),
                delta=f(0.0), ax=f(0.0), ay=f(0.0), slide=np.zeros(n, bool))


def forward_speed(st):
    return st["vx"] * np.cos(st["psi"]) + st["vy"] * np.sin(st["psi"])


def vehicle_step(sp, st, steer_cmd, a_cmd, mu, dt, abs_on=False, rec=None):
    n = st["x"].shape[0]
    steer_cmd = np.broadcast_to(np.asarray(steer_cmd, float), (n,))
    a_cmd = np.broadcast_to(np.asarray(a_cmd, float), (n,))
    c, s = np.cos(st["psi"]), np.sin(st["psi"])
    u = c * st["vx"] + s * st["vy"]
    v = -s * st["vx"] + c * st["vy"]
    r = st["r"]
    m = sp.mass

    # steering with rate limit
    steer_cmd = np.clip(steer_cmd, -sp.max_steer, sp.max_steer)
    st["delta"] += np.clip(steer_cmd - st["delta"], -STEER_RATE * dt, STEER_RATE * dt)
    delta = st["delta"]

    W = sp.wheels
    nw = W.shape[0]
    xi, yi = W[:, 0][None, :], W[:, 1][None, :]

    # wheel loads with load transfer (uses filtered body accelerations)
    Fz = (m * G * W[:, 5][None, :]
          + m * st["ax"][:, None] * sp.h / sp.wheelbase * sp.klong[None, :]
          + m * st["ay"][:, None] * sp.h * sp.klat[None, :])
    Fz = np.maximum(Fz, 0.03 * m * G / nw)

    # contact patch velocities in wheel axes
    vpx = u[:, None] - r[:, None] * yi
    vpy = v[:, None] + r[:, None] * xi
    dl = delta[:, None] * W[:, 2][None, :]
    cd, sd = np.cos(dl), np.sin(dl)
    vwx = vpx * cd + vpy * sd
    vwy = -vpx * sd + vpy * cd
    alpha = np.arctan2(vwy, np.abs(vwx) + EPS_ALPHA)

    mu_ = np.asarray(mu, float)
    mu_ = mu_[:, None] if mu_.ndim == 1 else mu_
    Fmax = mu_ * Fz

    a_brake = np.maximum(-a_cmd, 0.0)[:, None]
    a_drive = np.maximum(a_cmd, 0.0)[:, None]
    Bd = a_brake * m * W[:, 4][None, :]
    Dd = np.minimum(a_drive, sp.drive_max_g * G) * m * W[:, 3][None, :]
    locked = (Bd > Fmax) if not abs_on else np.zeros_like(Bd, bool)

    kcap = 0.92 if abs_on else 1.0
    Fx_r = np.clip(Dd - Bd * np.tanh(vwx / 0.3), -kcap * Fmax, kcap * Fmax)
    Fy_free = -Fmax * np.tanh(sp.c_alpha * sp.ca_mult[None, :] * alpha / mu_)
    Fy_r = Fy_free * np.sqrt(np.maximum(1.0 - (Fx_r / np.maximum(Fmax, 1e-6)) ** 2, 0.0))

    vmag = np.sqrt(vwx ** 2 + vwy ** 2 + EPS_SLIDE ** 2)
    Fx_s = -Fmax * vwx / vmag
    Fy_s = -Fmax * vwy / vmag
    Fxw = np.where(locked, Fx_s, Fx_r)
    Fyw = np.where(locked, Fy_s, Fy_r)

    Fbx = Fxw * cd - Fyw * sd
    Fby = Fxw * sd + Fyw * cd
    Fx = Fbx.sum(1)
    Fy = Fby.sum(1)
    Mz = (xi * Fby - yi * Fbx).sum(1)

    # resistances
    Fx = Fx - sp.crr * m * G * np.tanh(u / 0.5) - 0.5 * RHO * sp.cda * u * np.abs(u)

    # toppled body (isotropic sliding)
    if st["slide"].any():
        mub = sp.slide_mu
        vb = np.sqrt(u ** 2 + v ** 2 + EPS_SLIDE ** 2)
        Fsx = -mub * m * G * u / vb
        Fsy = -mub * m * G * v / vb
        Ms = -mub * m * G * 0.45 * np.tanh(r / 0.3)
        sl = st["slide"]
        Fx = np.where(sl, Fsx, Fx)
        Fy = np.where(sl, Fsy, Fy)
        Mz = np.where(sl, Ms, Mz)

    Fwx = Fx * c - Fy * s
    Fwy = Fx * s + Fy * c
    st["vx"] += Fwx / m * dt
    st["vy"] += Fwy / m * dt
    st["r"] += Mz / sp.izz * dt
    st["x"] += st["vx"] * dt
    st["y"] += st["vy"] * dt
    st["psi"] += st["r"] * dt
    k = min(dt / 0.05, 1.0)                        # 50 ms suspension lag
    st["ax"] += (Fx / m - st["ax"]) * k
    st["ay"] += (Fy / m - st["ay"]) * k

    if rec is not None:
        rec["marking"] = locked | ((np.abs(alpha) > 0.22) & (np.abs(u)[:, None] > 3.0))
        rec["locked"] = locked
        rec["wx"] = st["x"][:, None] + c[:, None] * xi - s[:, None] * yi
        rec["wy"] = st["y"][:, None] + s[:, None] * xi + c[:, None] * yi
