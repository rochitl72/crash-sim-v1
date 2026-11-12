"""Minimal end-to-end example: build a scene, simulate it, read the impact report, reconstruct it.
Run from the project root:  python -m examples.custom_scene"""
from crashlab import vehicles as V
from crashlab.drivers import accel_script, pursuit
from crashlab.engine import Scene, VehicleCfg, run_scene
from crashlab.scenes import road_straight
from crashlab.inverse import reconstruct

car, suv = V.sedan(), V.suv()
a = VehicleCfg("A", car, x=0, y=0, psi=0, speed=20.0,
               steer_fn=pursuit([(0, 0), (300, 0)], car.wheelbase),          # keep the lane
               accel_fn=accel_script([(0, "hold", 20.0), (2.8, "brake", 1.5)]))  # late, locked-wheel braking
b = VehicleCfg("B", suv, x=45, y=0, psi=0, speed=8.0,
               steer_fn=pursuit([(0, 0), (300, 0)], suv.wheelbase),
               accel_fn=accel_script([(0, "hold", 8.0)]))
scene = Scene("demo", "Demo rear-end", "", "", [a, b], road_straight(-20, 200, -3.5, 3.5), mu=0.75, t_end=10)

res = run_scene(scene)
for e in res["events"]:
    print(f"impact at t={e['t']:.2f}s  closing {e['closing_kmh']} km/h  "
          f"speeds {e['speed_a_kmh']}/{e['speed_b_kmh']} km/h  dV {e['dv_a_kmh']}/{e['dv_b_kmh']} km/h")

out, model, posterior = reconstruct(res, N=600)
for key in ("v_a", "v_b"):
    s = out["summary"][key]
    print(f"{key}: truth {s['truth']:.1f}  posterior median {s['p50']:.1f}  90% range {s['p5']:.1f}-{s['p95']:.1f} km/h")
