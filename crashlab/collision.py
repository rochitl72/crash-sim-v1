"""Collision detection (SAT on oriented rectangles) and impulse-momentum impact.

Impact model = classic reconstruction practice: conserve linear + angular momentum
through the contact, restitution coefficient e along the contact normal, Coulomb
friction mu_c along the tangent.
"""
import numpy as np


def corners(spec, st, i=0):
    p = st["psi"][i]
    cx = st["x"][i] + np.cos(p) * spec.rect_off
    cy = st["y"][i] + np.sin(p) * spec.rect_off
    hl, hw = spec.length / 2, spec.width / 2
    loc = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])          # CCW
    c, s = np.cos(p), np.sin(p)
    R = np.array([[c, -s], [s, c]])
    return loc @ R.T + np.array([cx, cy])


def _normals(poly):
    e = np.roll(poly, -1, axis=0) - poly
    nrm = np.stack([-e[:, 1], e[:, 0]], 1)
    return nrm / np.linalg.norm(nrm, axis=1, keepdims=True)


def _clip(subject, clipper):
    out = [p for p in subject]
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        inp, out = out, []
        if not inp:
            break
        s_pt = inp[-1]
        ins = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

        def inter(p, q):
            d1, d2 = q - p, b - a
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-12:
                return q
            t = ((a[0] - p[0]) * d2[1] - (a[1] - p[1]) * d2[0]) / den
            return p + t * d1

        for e_pt in inp:
            if ins(e_pt):
                if not ins(s_pt):
                    out.append(inter(s_pt, e_pt))
                out.append(e_pt)
            elif ins(s_pt):
                out.append(inter(s_pt, e_pt))
            s_pt = e_pt
    return out


def sat_contact(ca, cb):
    """Return (n, depth, P) with n pointing from A to B, or None if separated."""
    axes = np.vstack([_normals(ca), _normals(cb)])
    pa, pb = ca @ axes.T, cb @ axes.T
    ov = np.minimum(pa.max(0), pb.max(0)) - np.maximum(pa.min(0), pb.min(0))
    if (ov <= 0).any():
        return None
    k = int(np.argmin(ov))
    n = axes[k].copy()
    if n @ (cb.mean(0) - ca.mean(0)) < 0:
        n = -n
    poly = _clip(list(ca), cb)
    if len(poly) >= 3:
        P = np.mean(np.array(poly), axis=0)
    else:
        P = 0.5 * (ca.mean(0) + cb.mean(0))
    return n, float(ov[k]), P


def contact_velocity_terms(spA, A, spB, B, P, n):
    rax, ray = P[0] - A["x"], P[1] - A["y"]
    rbx, rby = P[0] - B["x"], P[1] - B["y"]
    vax, vay = A["vx"] - A["r"] * ray, A["vy"] + A["r"] * rax
    vbx, vby = B["vx"] - B["r"] * rby, B["vy"] + B["r"] * rbx
    vc = (vax - vbx) * n[0] + (vay - vby) * n[1]
    return rax, ray, rbx, rby, vc


def apply_impulse(spA, A, spB, B, P, n, e, mu_c):
    """Apply a frictional impact impulse in place. Arrays are (N,), P and n are (2,)
    or (N,2)-broadcastable scalars. Returns the normal impulse J (N,)."""
    mA, mB, IA, IB = spA.mass, spB.mass, spA.izz, spB.izz
    nx, ny = n[0], n[1]
    rax, ray, rbx, rby, vc = contact_velocity_terms(spA, A, spB, B, P, n)
    cA, cB = rax * ny - ray * nx, rbx * ny - rby * nx
    kn = 1 / mA + 1 / mB + cA ** 2 / IA + cB ** 2 / IB
    J = np.where(vc > 0, (1 + e) * vc / kn, 0.0)
    A["vx"] -= J * nx / mA
    A["vy"] -= J * ny / mA
    A["r"] -= cA * J / IA
    B["vx"] += J * nx / mB
    B["vy"] += J * ny / mB
    B["r"] += cB * J / IB
    # tangential (friction) impulse
    tx, ty = -ny, nx
    rax, ray, rbx, rby, _ = contact_velocity_terms(spA, A, spB, B, P, n)
    vax, vay = A["vx"] - A["r"] * ray, A["vy"] + A["r"] * rax
    vbx, vby = B["vx"] - B["r"] * rby, B["vy"] + B["r"] * rbx
    vt = (vax - vbx) * tx + (vay - vby) * ty
    dA, dB = rax * ty - ray * tx, rbx * ty - rby * tx
    kt = 1 / mA + 1 / mB + dA ** 2 / IA + dB ** 2 / IB
    Jt = np.clip(-vt / kt, -mu_c * J, mu_c * J)
    A["vx"] += Jt * tx / mA
    A["vy"] += Jt * ty / mA
    A["r"] += dA * Jt / IA
    B["vx"] -= Jt * tx / mB
    B["vy"] -= Jt * ty / mB
    B["r"] -= dB * Jt / IB
    return J


def kinetic_energy(sp, st):
    return 0.5 * sp.mass * (st["vx"] ** 2 + st["vy"] ** 2) + 0.5 * sp.izz * st["r"] ** 2
