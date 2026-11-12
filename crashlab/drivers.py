"""Scripted / reactive driver models. A driver is two callables:
    steer_fn(t, st, vehicles) -> steering angle at the wheels [rad]
    accel_fn(t, st, vehicles) -> longitudinal acceleration command [m/s^2] (negative = brake)
`st` is the vehicle's own state dict (batch size 1)."""
import numpy as np
from .vehicles import G


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _fwd(st):
    return float(st["vx"][0] * np.cos(st["psi"][0]) + st["vy"][0] * np.sin(st["psi"][0]))


def accel_script(segments):
    """segments: [(t_start, kind, value)], kind in hold|brake|coast|accel.
    hold  -> cruise at `value` m/s       brake -> decel of `value` g (>mu locks wheels)"""
    segs = sorted(segments, key=lambda s: s[0])

    def f(t, st, veh):
        cur = segs[0]
        for sg in segs:
            if t >= sg[0]:
                cur = sg
        _, kind, val = cur
        if kind == "hold":
            u = _fwd(st)
            return float(np.clip(2.0 * (val - u) + (0.25 if u > 2 else 0.0), -3.0, 2.0))
        if kind == "brake":
            return -val * G
        if kind == "accel":
            return float(val)
        return 0.0
    return f


def pursuit(points, wheelbase, ld_min=4.0, k=0.45):
    """Pure-pursuit path follower over a polyline of waypoints."""
    P = np.asarray(points, float)
    d = P[-1] - P[-2]
    P = np.vstack([P, P[-1] + d / np.linalg.norm(d) * 80.0])        # extend past the end
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    ss = np.arange(0, s[-1], 0.25)
    D = np.stack([np.interp(ss, s, P[:, 0]), np.interp(ss, s, P[:, 1])], 1)
    state = {"i": 0}

    def f(t, st, veh):
        x, y, psi = st["x"][0], st["y"][0], st["psi"][0]
        sp = float(np.hypot(st["vx"][0], st["vy"][0]))
        ld = max(ld_min, k * sp)
        i0 = state["i"]
        win = D[i0:i0 + 240]
        i = i0 + int(np.argmin(np.hypot(win[:, 0] - x, win[:, 1] - y)))
        state["i"] = i
        j = i
        while j < len(D) - 1 and np.hypot(D[j, 0] - x, D[j, 1] - y) < ld:
            j += 1
        eta = wrap(np.arctan2(D[j, 1] - y, D[j, 0] - x) - psi)
        return float(np.arctan2(2.0 * wheelbase * np.sin(eta), ld))
    return f


def reactive_follow(v_cruise, vis, reaction, brake_g=1.0):
    """Cruise; once any vehicle ahead (same lane) is inside `vis` metres, brake after
    `reaction` seconds. Models fog / poor-visibility pile-ups."""
    state = {"seen": None}
    cruise = accel_script([(0, "hold", v_cruise)])

    def f(t, st, veh):
        me = next(v for v in veh if v.st is st)
        if state["seen"] is None:
            best = 1e9
            for o in veh:
                if o is me:
                    continue
                dx, dy = o.st["x"][0] - st["x"][0], o.st["y"][0] - st["y"][0]
                c, s = np.cos(st["psi"][0]), np.sin(st["psi"][0])
                lon, lat = dx * c + dy * s, -dx * s + dy * c
                if lon > 0 and abs(lat) < (me.spec.width + o.spec.width) / 2 + 0.3:
                    best = min(best, lon - (me.spec.length + o.spec.length) / 2)
            if best < vis:
                state["seen"] = t
        if state["seen"] is not None and t >= state["seen"] + reaction:
            return -brake_g * G
        return cruise(t, st, veh)
    return f
