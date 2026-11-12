"""Build everything: run the six scenes, reconstruct the two-vehicle ones, run validation
checks, export dist/data.json and bundle dist/crashlab.html (single self-contained file).

This is v1: a 2D-only build. No damage model, no occupant/airbag data, no 3D —
just the rigid-body physics, the SAT collision/impulse model and the Monte Carlo
rest-position reconstruction, rendered on a flat top-down canvas.
"""
import json, os, sys, time
import numpy as np
from crashlab import scenes, inverse
from crashlab.engine import run_scene
from tests import test_physics as TP

RECON = {"rear_end": 21, "tbone": 7, "head_on": 5, "two_wheeler": 3}


def scene_payload(res):
    sc = res["scene"]
    return dict(
        id=sc.id, title=sc.title, subtitle=sc.subtitle, story=sc.story, tags=list(sc.tags),
        view=sc.view, road=dict(polys=[[list(map(float, p)) for p in poly] for poly in sc.road["polys"]],
                                lines=[dict(pts=[list(map(float, p)) for p in ln["pts"]], dash=ln["dash"],
                                            color=ln["color"], w=ln["w"]) for ln in sc.road["lines"]]),
        veh=[dict(label=v.cfg.label, note=v.cfg.note, abs=bool(v.cfg.abs_on), spec=v.spec.meta(),
                  speed0=v.cfg.speed * 3.6) for v in res["veh"]],
        frames=res["frames"], dt=res["dt"],
        t0=round(max(0.0, (res["events"][0]["t"] if res["events"] else 0.0) - 2.2), 2),
        events=[{k: v for k, v in e.items() if k not in ("pre", "Pf", "nf", "ia", "ib", "t0")} |
                dict(ia=e["ia"], ib=e["ib"]) for e in res["events"]],
        marks=res["marks"], final=res["final"], mu=sc.mu, e=sc.e, recon=None)


def validation(models):
    chk = []
    sk = TP.t_skid()
    worst = max(abs(r[3]) for r in sk)
    chk.append(dict(name="Skid distance vs textbook v²/(2μg)",
                    detail="20 m/s, μ=0.8, locked wheels; sedan / SUV / truck / motorcycle",
                    got="%.1f %% max deviation" % worst, expect="a few %, sign = shorter (rolling + aero drag)",
                    ok=bool(worst < 6)))
    m = TP.t_momentum()
    chk.append(dict(name="Momentum conservation through impact",
                    detail="oblique impact with friction and restitution (linear + angular)",
                    got="Δp = %.1e kg·m/s, ΔL = %.1e" % (m["dp"], m["dL"]), expect="≈ 0 (machine precision)",
                    ok=bool(m["dp"] < 1e-6 and m["dL"] < 1e-6)))
    p = TP.t_plastic()
    chk.append(dict(name="Perfectly plastic impact energy loss",
                    detail="central head-on, e=0: ΔE = ½·μ_red·v_closing²",
                    got="%.3f kJ (theory %.3f kJ)" % (p["loss_sim"], p["loss_theory"]), expect="equal",
                    ok=bool(abs(p["loss_sim"] - p["loss_theory"]) < 1e-3)))
    el = TP.t_elastic()
    chk.append(dict(name="Elastic impact, equal masses",
                    detail="e=1 central head-on: velocities swap",
                    got="A: +12 → %.1f, B: −6 → %.1f m/s" % (el["A_after"], el["B_after"]), expect="A: −6, B: +12",
                    ok=bool(abs(el["A_after"] + 6) < 1e-9 and abs(el["B_after"] - 12) < 1e-9)))
    dc = TP.t_dt_convergence()
    dpos = abs(dc[0][1] - dc[2][1]) + abs(dc[0][2] - dc[2][2])
    chk.append(dict(name="Time-step convergence",
                    detail="4 ms vs 1 ms steps, 3 s manoeuvre with ABS braking and steering",
                    got="%.2f m position difference over %.0f m" % (dpos, dc[2][1]), expect="< 0.5 % of distance",
                    ok=bool(dpos / dc[2][1] < 0.005)))
    for sid, (model, th) in models.items():
        out = model.simulate(th)[0]
        dxy = max(np.hypot(out[0] - model.truth_rest[0], out[1] - model.truth_rest[1]),
                  np.hypot(out[3] - model.truth_rest[3], out[4] - model.truth_rest[4]))
        chk.append(dict(name="Solver forward model vs full scene · %s" % sid,
                        detail="post-impact model at the true speeds vs full multi-vehicle simulation",
                        got="%.2f m max rest-position difference" % dxy, expect="< 0.3 m", ok=bool(dxy < 0.3)))
    return chk


def main(data_only=False):
    t0 = time.time()
    payload, models = [], {}
    for fn in scenes.ALL:
        ts = time.time()
        res = run_scene(fn())
        p = scene_payload(res)
        sid = p["id"]
        if sid in RECON:
            out, model, th = inverse.reconstruct(res, N=1200, seed=RECON[sid])
            out.pop("scene", None)
            p["recon"] = out
            models[sid] = (model, np.array([[model.v_true[0], model.v_true[1], model.mu_true, model.e_true]]))
            ins = [v["inside90"] for k, v in out["summary"].items() if v]
            print(f"  {sid}: events={len(p['events'])} recon {out['seconds']}s  truth inside 90%: {sum(ins)}/{len(ins)}")
        else:
            print(f"  {sid}: events={len(p['events'])}")
        payload.append(p)
        print(f"    ({time.time() - ts:.1f}s)")
    from crashlab import vehicles as V
    catalogue = []
    for f in V.CATALOGUE:
        sp = f()
        m = sp.meta(); m.pop("wheels", None)
        m.update(wheelbase=sp.wheelbase)
        catalogue.append(m)
    data = dict(scenes=payload, validation=validation(models), catalogue=catalogue,
                meta=dict(built=time.strftime("%Y-%m-%d"), seconds=round(time.time() - t0, 1)))

    def clean(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(type(o))
    os.makedirs("dist", exist_ok=True)
    js = json.dumps(data, default=clean, separators=(",", ":"))
    open("dist/data.json", "w", encoding="utf-8").write(js)
    print(f"data.json {len(js) / 1e6:.2f} MB  ({time.time() - t0:.0f}s)")
    if not data_only and os.path.exists("viewer/template.html"):
        bundle(js)


HERE = os.path.dirname(os.path.abspath(__file__))


def bundle_html(js):
    html = open(os.path.join(HERE, "viewer/template.html"), encoding="utf-8").read()
    return html.replace("/*__DATA__*/null", js.replace("</", "<\\/"), 1)


def bundle(js):
    html = bundle_html(js)
    open("dist/crashlab.html", "w", encoding="utf-8").write(html)
    print(f"dist/crashlab.html {len(html) / 1e6:.2f} MB")


if __name__ == "__main__":
    if "--html-only" in sys.argv:
        bundle(open("dist/data.json", encoding="utf-8").read())
    else:
        main(data_only="--data-only" in sys.argv)
