"""Probabilistic crash reconstruction.

Unknowns  theta = [v_A, v_B, mu, e]
   v_A, v_B : impact speeds along each vehicle's heading at first contact [m/s]
   mu       : tyre-road friction (shared)
   e        : restitution coefficient of the impact
Evidence  rest poses (x, y, heading) of both vehicles, pre-impact skid lengths,
          a drag-sled friction measurement (prior on mu).
Method    forward model = impulse-momentum impact + 4-wheel post-impact motion,
          vectorised over thousands of candidate theta at once;
          posterior = tempered Sequential Monte Carlo (adaptive tempering, resample-move).
Output    posterior over speeds (not one number), plus the derived speed at brake onset.
"""
import time
import numpy as np
from .vehicles import G
from .dynamics import new_state, vehicle_step
from .collision import apply_impulse
from .drivers import wrap

# default noise assumptions (documented in the viewer)
MEAS_XY, MEAS_PSI = 0.25, np.radians(2.0)          # survey error of the rest positions
SIG_XY, SIG_PSI = 0.60, np.radians(6.0)             # likelihood width = survey + model error
MU_SIGMA = 0.06                                     # drag-sled measurement uncertainty
LO = np.array([1.0, 1.0, 0.40, 0.00])
HI = np.array([35.0, 35.0, 0.95, 0.50])


class PostImpactModel:
    def __init__(self, res, ev_index=0, dt=0.005, t_max=8.0):
        sc = res["scene"]
        ev = res["events"][ev_index]
        self.ev, self.scene = ev, sc
        va, vb = res["veh"][ev["ia"]], res["veh"][ev["ib"]]
        self.va, self.vb = va, vb
        self.sa, self.sb = va.spec, vb.spec
        self.pa, self.pb = ev["pre"]["a"], ev["pre"]["b"]
        self.P, self.n = np.array(ev["Pf"]), np.array(ev["nf"])
        self.mu_c, self.dt, self.t_max = sc.mu_c, dt, t_max
        self.mu_true, self.e_true = sc.mu, sc.e
        self.aA = -1.5 * G if va.cfg.post_impact == "brake" else 0.0
        self.aB = -1.5 * G if vb.cfg.post_impact == "brake" else 0.0
        self.absA, self.absB = va.cfg.abs_on, vb.cfg.abs_on
        self.truth_rest = np.array([res["final"][ev["ia"]]["x"], res["final"][ev["ia"]]["y"],
                                    res["final"][ev["ia"]]["psi"], res["final"][ev["ib"]]["x"],
                                    res["final"][ev["ib"]]["y"], res["final"][ev["ib"]]["psi"]])
        # pre-impact skid lengths (longest wheel mark before first contact)
        self.skid = {}
        for who, v, idx in (("a", va, ev["ia"]), ("b", vb, ev["ib"])):
            best = 0.0
            for m in res["marks"]:
                if m["v"] != idx:
                    continue
                pts = [p for p in m["pts"] if p[2] < ev["t0"] - 0.02]
                if len(pts) > 1:
                    L = sum(np.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
                            for i in range(len(pts) - 1))
                    best = max(best, L)
            self.skid[who] = best if (best > 1.0 and not v.cfg.abs_on) else None
        self.v_true = np.array([np.hypot(self.pa["vx"], self.pa["vy"]), np.hypot(self.pb["vx"], self.pb["vy"])])
        self.v0_true = np.array([va.cfg.speed, vb.cfg.speed])

    # ---------------------------------------------------------------- forward model
    def simulate(self, theta):
        theta = np.atleast_2d(theta)
        N = theta.shape[0]
        vA, vB, mu, e = theta.T
        pa, pb = self.pa, self.pb
        A = new_state(N, pa["x"], pa["y"], pa["psi"], vA)
        B = new_state(N, pb["x"], pb["y"], pb["psi"], vB)
        apply_impulse(self.sa, A, self.sb, B, self.P, self.n, e, self.mu_c)
        if self.sa.slide_on_hit:
            A["slide"][:] = True
        if self.sb.slide_on_hit:
            B["slide"][:] = True
        for k in range(int(self.t_max / self.dt)):
            vehicle_step(self.sa, A, A["delta"], self.aA, mu, self.dt, self.absA)
            vehicle_step(self.sb, B, B["delta"], self.aB, mu, self.dt, self.absB)
            if k % 50 == 49:
                mv = max(np.hypot(A["vx"], A["vy"]).max() + np.abs(A["r"]).max(),
                         np.hypot(B["vx"], B["vy"]).max() + np.abs(B["r"]).max())
                if mv < 0.05:
                    break
        return np.stack([A["x"], A["y"], A["psi"], B["x"], B["y"], B["psi"]], 1)

    def loglik(self, theta, obs):
        theta = np.atleast_2d(theta)
        inb = ((theta >= LO) & (theta <= HI)).all(1)
        th = np.clip(theta, LO, HI)
        out = self.simulate(th)
        rxy = (out[:, [0, 1, 3, 4]] - obs[[0, 1, 3, 4]]) / SIG_XY
        rps = wrap(out[:, [2, 5]] - obs[[2, 5]]) / SIG_PSI
        ll = -0.5 * ((rxy ** 2).sum(1) + (rps ** 2).sum(1))
        ll += -0.5 * ((th[:, 2] - self.mu_meas) / MU_SIGMA) ** 2      # drag-sled measurement
        return np.where(inb, ll, -np.inf), out


def _systematic(w, rng):
    N = len(w)
    pos = (rng.random() + np.arange(N)) / N
    return np.minimum(np.searchsorted(np.cumsum(w), pos), N - 1)


def smc(model, obs, N=1000, ess_frac=0.5, n_move=3, rng=None, verbose=False):
    rng = rng or np.random.default_rng(0)
    d = 4
    th = LO + (HI - LO) * rng.random((N, d))
    th[:, 2] = np.clip(model.mu_meas + MU_SIGMA * rng.standard_normal(N), LO[2], HI[2])   # prior ~ drag-sled
    ll, _ = model.loglik(th, obs)
    ll = np.where(np.isfinite(ll), ll, -1e300)
    beta, stages = 0.0, 0
    while beta < 1.0:
        lo_b, hi_b = beta, 1.0
        def ess(b):
            lw = (b - beta) * ll
            w = np.exp(lw - lw.max()); w /= w.sum()
            return 1.0 / (w ** 2).sum()
        if ess(1.0) >= ess_frac * N:
            nb = 1.0
        else:
            for _ in range(40):
                mid = 0.5 * (lo_b + hi_b)
                if ess(mid) < ess_frac * N: hi_b = mid
                else: lo_b = mid
            nb = max(lo_b, beta + 1e-9)
        lw = (nb - beta) * ll
        w = np.exp(lw - lw.max()); w /= w.sum()
        idx = _systematic(w, rng)
        th, ll = th[idx], ll[idx]
        beta = nb
        cov = np.cov(th.T) + 1e-9 * np.eye(d)
        L = np.linalg.cholesky(cov)
        scale = 2.38 / np.sqrt(d) * 0.7
        acc = 0
        for _ in range(n_move):
            prop = th + scale * (rng.standard_normal((N, d)) @ L.T)
            llp, _ = model.loglik(prop, obs)
            llp = np.where(np.isfinite(llp), llp, -1e300)
            a = np.log(rng.random(N)) < beta * (llp - ll)
            th[a], ll[a] = prop[a], llp[a]
            acc += a.mean()
        stages += 1
        if verbose:
            print(f"    stage {stages:2d} beta={beta:.4f} accept={acc / n_move:.2f}")
        if stages > 60:
            break
    return th, ll, stages


def derive_v0(model, th, rng):
    """Speed at brake onset from impact speed + measured pre-impact skid length."""
    out = {}
    for k, who in enumerate(("a", "b")):
        d = model.skid[who]
        if d is None:
            out[who] = None
            continue
        dm = np.maximum(d + MEAS_XY * 2 * rng.standard_normal(len(th)), 0.0)       # skid length survey error
        a_eff = th[:, 2] * G + 0.25
        out[who] = np.sqrt(th[:, k] ** 2 + 2 * a_eff * dm)
    return out


def summarize(x, truth=None, scale=1.0):
    x = np.asarray(x) * scale
    q = np.percentile(x, [5, 50, 95])
    s = dict(mean=float(x.mean()), p5=float(q[0]), p50=float(q[1]), p95=float(q[2]), sd=float(x.std()))
    if truth is not None:
        s["truth"] = float(truth * scale)
        s["inside90"] = bool(q[0] <= truth * scale <= q[2])
    return s


def reconstruct(res, ev_index=0, N=1000, seed=7, verbose=False, grid=36, n_ghost=36):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    model = PostImpactModel(res, ev_index)
    # evidence = true rest poses + survey noise (this is what a field investigator would measure)
    obs = model.truth_rest.copy()
    obs[[0, 1, 3, 4]] += MEAS_XY * rng.standard_normal(4)
    obs[[2, 5]] += MEAS_PSI * rng.standard_normal(2)
    model.mu_meas = model.mu_true + 0.04 * rng.standard_normal()           # drag-sled reading
    th, ll, stages = smc(model, obs, N=N, rng=rng, verbose=verbose)
    v0 = derive_v0(model, th, rng)

    # best single fit (what a conventional one-answer tool would report)
    best = th[np.argmax(ll)]

    # misfit surface over the two speeds with mu, e at their posterior medians
    mu_m, e_m = np.median(th[:, 2]), np.median(th[:, 3])
    gv = np.linspace(max(LO[0], min(th[:, 0].min(), model.v_true[0]) * 0.55), min(HI[0], max(th[:, 0].max(), model.v_true[0]) * 1.7), grid)
    gw = np.linspace(max(LO[1], min(th[:, 1].min(), model.v_true[1]) * 0.55), min(HI[1], max(th[:, 1].max(), model.v_true[1]) * 1.7), grid)
    GA, GB = np.meshgrid(gv, gw)
    gth = np.stack([GA.ravel(), GB.ravel(), np.full(GA.size, mu_m), np.full(GA.size, e_m)], 1)
    gll, _ = model.loglik(gth, obs)
    gll = gll.reshape(GA.shape)
    surf = np.exp(gll - np.nanmax(gll[np.isfinite(gll)]))

    # ghost rest positions from posterior draws
    gi = rng.choice(len(th), n_ghost, replace=False)
    ghosts = model.simulate(th[gi])

    kmh = 3.6
    out = dict(
        scene=res["scene"].id, event=ev_index + 1, N=N, stages=stages, seconds=round(time.time() - t0, 1),
        a=model.va.cfg.label, b=model.vb.cfg.label,
        obs=[float(v) for v in obs],
        truth=dict(v_imp_a=float(model.v_true[0] * kmh), v_imp_b=float(model.v_true[1] * kmh),
                   v0_a=float(model.v0_true[0] * kmh), v0_b=float(model.v0_true[1] * kmh),
                   mu=model.mu_true, e=model.e_true),
        summary=dict(v_a=summarize(th[:, 0], model.v_true[0], kmh), v_b=summarize(th[:, 1], model.v_true[1], kmh),
                     mu=summarize(th[:, 2], model.mu_true), e=summarize(th[:, 3], model.e_true),
                     v0_a=summarize(v0["a"], model.v0_true[0], kmh) if v0["a"] is not None else None,
                     v0_b=summarize(v0["b"], model.v0_true[1], kmh) if v0["b"] is not None else None),
        best=dict(v_a=float(best[0] * kmh), v_b=float(best[1] * kmh), mu=float(best[2]), e=float(best[3])),
        skid=dict(a=model.skid["a"], b=model.skid["b"]),
        samples=dict(v_a=[round(float(x * kmh), 2) for x in th[::3, 0]],
                     v_b=[round(float(x * kmh), 2) for x in th[::3, 1]],
                     mu=[round(float(x), 3) for x in th[::3, 2]],
                     e=[round(float(x), 3) for x in th[::3, 3]],
                     v0_a=[round(float(x * kmh), 2) for x in v0["a"][::3]] if v0["a"] is not None else None,
                     v0_b=[round(float(x * kmh), 2) for x in v0["b"][::3]] if v0["b"] is not None else None),
        surface=dict(va=[round(float(x * kmh), 2) for x in gv], vb=[round(float(x * kmh), 2) for x in gw],
                     z=[[round(float(z), 4) for z in row] for row in surf]),
        ghosts=[[round(float(x), 3) for x in g] for g in ghosts],
        corr=float(np.corrcoef(th[:, 0], th[:, 1])[0, 1]),
    )
    return out, model, th
