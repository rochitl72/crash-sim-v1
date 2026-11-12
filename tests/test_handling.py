"""Steady-state handling sanity: constant speed + constant steer -> skidpad.
Checks (a) lateral accel ~ kinematic estimate at low demand, (b) understeer sign,
(c) no spin-out when steering demand exceeds the friction limit (should plough, not spin)."""
import numpy as np
from crashlab import vehicles as V
from crashlab.dynamics import new_state, vehicle_step, forward_speed

G = 9.81


def run(spec, speed, steer, mu=0.8, T=6.0, dt=0.002, hold=True):
    st = new_state(1, 0, 0, 0, speed)
    for k in range(int(T / dt)):
        u = forward_speed(st)[0]
        a = 2.0 * (speed - u) + 0.25
        vehicle_step(spec, st, steer if k * dt > 0.3 else 0.0, a, mu, dt, False)
    u = forward_speed(st)[0]
    v = -np.sin(st["psi"][0]) * st["vx"][0] + np.cos(st["psi"][0]) * st["vy"][0]
    return u, st["r"][0], st["ay"][0] / G, np.degrees(np.arctan2(v, u))


if __name__ == "__main__":
    sp = V.sedan()
    L = sp.wheelbase
    print("sedan, mu=0.8  | speed  steer  ->  u   yaw-rate  ay[g]  sideslip[deg]  | kinematic r, ay")
    for spd, d in [(10, 0.03), (15, 0.03), (20, 0.03), (20, 0.06), (25, 0.06), (25, 0.10)]:
        u, r, ay, beta = run(sp, spd, d)
        print(f"  {spd:5.1f} {d:6.3f}  -> {u:5.1f} {r:8.3f} {ay:7.2f} {beta:9.1f}   | {spd*d/L:6.3f} {spd*spd*d/L/G:5.2f}g")
    for nm, sp2 in [("suv", V.suv()), ("truck", V.truck()), ("hatch", V.hatchback())]:
        u, r, ay, beta = run(sp2, 18, 0.04)
        print(f"{nm:6s} 18 m/s, 0.04 rad -> yaw {r:.3f}  ay {ay:.2f}g  beta {beta:.1f}deg")
