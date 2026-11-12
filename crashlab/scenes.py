"""Six crash scenes (the original set). World frame: x east, y north (metres),
left-hand traffic (India). Heading +x -> its left is +y, so an eastbound
vehicle keeps to the +y half."""
import numpy as np
from . import vehicles as V
from .drivers import accel_script, pursuit, reactive_follow
from .engine import Scene, VehicleCfg

KMH = 1 / 3.6


# ------------------------------------------------------------------ road helpers
def road_straight(x0, x1, y0, y1, lanes=(), edge=True, centre_dash=True):
    polys = [[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]
    lines = []
    for yl in lanes:
        lines.append(dict(pts=[(x0, yl), (x1, yl)], dash=[3, 6], color="#e8e8e8", w=0.14))
    if edge:
        lines += [dict(pts=[(x0, y0 + 0.35), (x1, y0 + 0.35)], dash=None, color="#e8e8e8", w=0.16),
                  dict(pts=[(x0, y1 - 0.35), (x1, y1 - 0.35)], dash=None, color="#e8e8e8", w=0.16)]
    return dict(polys=polys, lines=lines)


def road_cross(ext, hw=3.5, stop_lines=True, zebra=False):
    """Plus-shaped junction centred on the origin; arms of length ext, half-width hw."""
    polys = [[(-ext, -hw), (ext, -hw), (ext, hw), (-ext, hw)],
             [(-hw, -ext), (hw, -ext), (hw, ext), (-hw, ext)]]
    lines = []
    for sgn in (-1, 1):
        lines.append(dict(pts=[(sgn * hw, 0), (sgn * ext, 0)], dash=[3, 5], color="#f2c94c", w=0.14))
        lines.append(dict(pts=[(0, sgn * hw), (0, sgn * ext)], dash=[3, 5], color="#f2c94c", w=0.14))
    if stop_lines:
        for (a, b) in [((-hw - 1.2, 0.1), (-hw - 1.2, hw)), ((hw + 1.2, -hw), (hw + 1.2, -0.1)),
                       ((-hw, -hw - 1.2), (-0.1, -hw - 1.2)), ((0.1, hw + 1.2), (hw, hw + 1.2))]:
            lines.append(dict(pts=[a, b], dash=None, color="#e8e8e8", w=0.45))
    return dict(polys=polys, lines=lines)


def plan(v0, skid, T_imp, a_eff):
    """Brake planning. A vehicle cruises at v0, brakes at t_brake with deceleration a_eff over
    `skid` metres and reaches the impact point at T_imp. Returns (t_brake, v_imp, travel)."""
    v_imp = float(np.sqrt(max(v0 ** 2 - 2 * a_eff * skid, 0.0)))
    t_skid = (v0 - v_imp) / a_eff
    t_brake = T_imp - t_skid
    return t_brake, v_imp, v0 * t_brake + skid


def a_lock(mu):      # locked-wheel deceleration incl. rolling + aero drag (typical car)
    return mu * 9.81 + 0.25


def a_abs(mu):
    return 0.92 * mu * 9.81 + 0.25


def solo_time(cfg, cond, t_end=10.0):
    """Run one vehicle alone; return (t, frame) when cond(frame) first holds."""
    from .engine import run_scene
    r = run_scene(Scene("solo", "", "", "", [cfg], dict(polys=[], lines=[]), t_end=t_end, t_min=99))
    for i, fr in enumerate(r["frames"]):
        if cond(fr[0]):
            return i * r["dt"], fr[0]
    raise RuntimeError("condition never met")


# ------------------------------------------------------------------ S1 rear-end
def s1_rear_end():
    sedan, hatch = V.sedan("#2ec4b6"), V.hatchback("#ffb703")
    vA0, vB = 22.0, 9.0
    xB0, tb = 26.2, 1.0
    A = VehicleCfg("A", sedan, 0.0, 1.05, 0.0, vA0,
                   pursuit([(0, 1.05), (300, 1.05)], sedan.wheelbase),
                   accel_script([(0, "hold", vA0), (tb, "brake", 1.5)]),
                   note="Distracted driver, late braking, no ABS")
    B = VehicleCfg("B", hatch, xB0, 1.75, 0.0, vB,
                   pursuit([(0, 1.75), (300, 1.75)], hatch.wheelbase),
                   accel_script([(0, "hold", vB)]),
                   note="Slow-moving hatchback, constant 32 km/h")
    return Scene("rear_end", "Rear-end on a national highway",
                 "Late braking · 60% lateral overlap · yaw after impact",
                 "A sedan closes on a slow hatchback in the left lane. The driver notices late, "
                 "locks the brakes for ~17 m and strikes the rear of the hatchback with 60% overlap.",
                 [A, B], road_straight(-30, 130, -3.5, 3.5, lanes=[0]), mu=0.75, e=0.20,
                 t_end=12, t_min=2.5, view=[-8, -7.5, 66, 7.5], tags=["Rear-end", "Skid marks", "Offset"])


# ------------------------------------------------------------------ S2 T-bone
def s2_tbone():
    sedan, suv = V.sedan("#2ec4b6"), V.suv("#8ecae6")
    vA0, vB0, mu, T = 14.0, 11.5, 0.75, 1.8
    xa_imp, yb_imp = -5.0, 1.3
    tbA, _, trA = plan(vA0, 5.2, T, a_lock(mu))
    tbB, _, trB = plan(vB0, 3.6, T, a_lock(mu))
    A = VehicleCfg("A", sedan, xa_imp - trA, 1.75, 0.0, vA0,
                   pursuit([(-200, 1.75), (200, 1.75)], sedan.wheelbase),
                   accel_script([(0, "hold", vA0), (tbA, "brake", 1.5)]),
                   note="Enters the junction at ~50 km/h, brakes late")
    B = VehicleCfg("B", suv, -1.75, yb_imp - trB, np.pi / 2, vB0,
                   pursuit([(-1.75, -200), (-1.75, 200)], suv.wheelbase),
                   accel_script([(0, "hold", vB0), (tbB, "brake", 1.5)]),
                   note="Does not give way; brakes ~3.6 m before impact")
    return Scene("tbone", "T-bone at an uncontrolled junction",
                 "Side impact · both vehicles skid · SUV rotates",
                 "Neither driver gives way at an unsigned crossing. The sedan brakes for ~5 m and its front "
                 "strikes the left flank of the SUV, which rotates as it is pushed across the junction.",
                 [A, B], road_cross(120), mu=0.75, e=0.20, t_end=12, t_min=2.5,
                 view=[-24, -14, 24, 14], tags=["Side impact", "Junction", "Two skid sets"])


# ------------------------------------------------------------------ S3 head-on
def s3_head_on():
    suv, sed = V.suv("#8ecae6"), V.sedan("#ff7f50")
    vA0, vC0, mu, T = 21.0, 18.0, 0.70, 2.4
    tbA, _, trA = plan(vA0, 8.0, T, a_abs(mu))
    tbC, _, trC = plan(vC0, 6.0, T, a_lock(mu))
    xA, xC = -2.275, 2.175
    A = VehicleCfg("A", suv, xA - trA, -1.75, 0.0, vA0,
                   pursuit([(-200, -1.75), (xA - 22, -1.75), (xA, -0.95), (xA + 22, 0.4), (xA + 80, 1.75)],
                           suv.wheelbase, ld_min=7.0),
                   accel_script([(0, "hold", vA0), (tbA, "brake", 0.8)]),
                   abs_on=True, note="Overtaking in the wrong lane, ABS braking + swerve, no skid marks")
    C = VehicleCfg("C", sed, xC + trC, -1.75, np.pi, vC0,
                   pursuit([(200, -1.75), (-200, -1.75)], sed.wheelbase),
                   accel_script([(0, "hold", vC0), (tbC, "brake", 1.5)]),
                   note="Oncoming sedan, no ABS, locks wheels and skids straight")
    return Scene("head_on", "Offset head-on during overtaking",
                 "ABS swerve (no marks) vs locked-wheel skid · ~50% overlap",
                 "An SUV overtakes in the oncoming lane. Both drivers see each other late; the SUV swerves back "
                 "with ABS braking (leaves no skid marks), the sedan locks up and skids straight. They meet "
                 "with roughly half overlap.",
                 [A, C], road_straight(-90, 110, -3.5, 3.5, lanes=[0]), mu=0.70, e=0.25,
                 t_end=12, t_min=2.5, view=[-26, -8, 26, 8], tags=["Head-on", "Overtaking", "ABS vs no-ABS"])


# ------------------------------------------------------------------ S4 two-wheeler
def s4_motorcycle(ycond=-1.0, xoff=3.3):
    sed, bike_spec = V.sedan("#2ec4b6"), V.motorcycle("#e5383b")
    R = 8.0
    cx, cy = -6.0, 1.75 - R
    th = np.linspace(0, np.pi / 2, 24)
    arc = [(cx + R * np.sin(t), cy + R * np.cos(t)) for t in th]
    path = [(-60, 1.75), (-6, 1.75)] + arc[1:] + [(2.0, -40)]
    vA = 5.0

    def make_car():
        return VehicleCfg("A", sed, -22.0, 1.75, 0.0, vA,
                          pursuit(path, sed.wheelbase, ld_min=4.5, k=0.3),
                          accel_script([(0, "hold", vA)]),
                          note="Right-turn across oncoming traffic at ~18 km/h")
    A = make_car()
    mu = 0.80
    Tc, fr = solo_time(make_car(), lambda f: f[1] < ycond)
    xc = fr[0] + xoff
    vM0 = 15.5
    tbM, _, trM = plan(vM0, 7.0, Tc, a_lock(mu))

    def make_bike(x0):
        return VehicleCfg("M", bike_spec, x0, -1.75, np.pi, vM0,
                          pursuit([(200, -1.75), (-200, -1.75)], bike_spec.wheelbase),
                          accel_script([(0, "hold", vM0), (tbM, "brake", 0.85)]),
                          note="Oncoming two-wheeler at ~56 km/h, late braking")
    x0 = xc + trM
    for _ in range(5):
        t_arr, _ = solo_time(make_bike(x0), lambda f: f[0] < xc, t_end=8.0)
        x0 += (Tc - t_arr) * 12.0
    M = make_bike(x0)
    return Scene("two_wheeler", "Motorcycle vs right-turning car",
                 "Two-wheeler · rider slide · turning conflict",
                 "A sedan turns right across the oncoming lane. A motorcycle approaches at ~56 km/h, brakes "
                 "late and strikes the car's flank; the toppled bike then slides along the road.",
                 [A, M], road_cross(90), mu=0.80, e=0.15, t_end=12, t_min=2.5,
                 view=[-22, -12, 22, 12], tags=["Two-wheeler", "Turning conflict", "Rider slide"])


# ------------------------------------------------------------------ S5 fog pile-up
def s5_fog():
    T = V.truck("#e76f51")
    cars = [V.sedan("#2ec4b6"), V.suv("#8ecae6"), V.hatchback("#ffb703"), V.sedan("#c77dff")]
    vc = [22.0, 22.5, 23.0, 22.0]
    xs = [-62.0, -100.0, -140.0, -182.0]
    ys = [-1.55, -2.05, -1.60, -1.90]
    vis, rt = 26.0, 1.0
    veh = [VehicleCfg("T", T, 0.0, -1.75, 0.0, 6.0,
                      pursuit([(-300, -1.75), (300, -1.75)], T.wheelbase),
                      accel_script([(0, "hold", 6.0)]), note="Slow truck, hazards on, ~22 km/h")]
    for k, (sp, v0, x0, y0) in enumerate(zip(cars, vc, xs, ys)):
        veh.append(VehicleCfg("V%d" % (k + 1), sp, x0, y0, 0.0, v0,
                              pursuit([(-300, y0), (300, y0)], sp.wheelbase),
                              reactive_follow(v0, vis, rt, 1.5),
                              note="Fog: sees the vehicle ahead only at ~26 m, 1.0 s reaction"))
    return Scene("fog_pileup", "Fog pile-up on an expressway",
                 "5 vehicles · chain collisions · visibility 26 m",
                 "Dense fog. A slow truck ahead is invisible until ~26 m. Each driver reacts after one "
                 "second and locks the brakes; the leading car hits the truck and the followers pile in.",
                 veh, road_straight(-260, 140, -5.25, 5.25, lanes=[-1.75, 1.75]), mu=0.60, e=0.15,
                 t_end=20, t_min=4, view=[-48, -8, 46, 8], tags=["Multi-vehicle", "Fog", "Chain reaction"])


# ------------------------------------------------------------------ S6 compound
def s6_compound():
    bus_spec, suv, auto = V.truck("#f4a261"), V.suv("#8ecae6"), V.motorcycle("#e9c46a")
    vB, vS, mu, T = 15.5, 11.0, 0.75, 2.2
    tbB, _, trB = plan(vB, 6.0, T, a_lock(mu))
    B = VehicleCfg("B", bus_spec, -1.75, -4.975 - trB, np.pi / 2, vB,
                   pursuit([(-1.75, -200), (-1.75, 200)], bus_spec.wheelbase),
                   accel_script([(0, "hold", vB), (tbB, "brake", 1.5)]),
                   note="Bus runs the red light at ~56 km/h")
    S = VehicleCfg("S", suv, -1.5 - vS * T, 1.75, 0.0, vS,
                   pursuit([(-200, 1.75), (200, 1.75)], suv.wheelbase),
                   accel_script([(0, "hold", vS)]),
                   note="SUV crossing on green at ~40 km/h")
    R = VehicleCfg("R", auto, 1.75, 9.0, -np.pi / 2, 0.0,
                   pursuit([(1.75, 100), (1.75, -100)], auto.wheelbase),
                   accel_script([(0, "brake", 1.5)]),
                   note="Waiting at the stop line, parking brake on")
    return Scene("compound", "Compound crash at a signalised junction",
                 "Red-light runner · secondary impact · bystander vehicle",
                 "A bus runs the red light and hits the right flank of a crossing SUV. The SUV is thrown "
                 "north-east, rotating, into a two-wheeler waiting at the stop line.",
                 [B, S, R], road_cross(100), mu=0.75, e=0.20, t_end=14, t_min=3,
                 view=[-26, -13, 26, 24], tags=["Compound", "Secondary impact", "Heavy vehicle"])


ALL = [s1_rear_end, s2_tbone, s3_head_on, s4_motorcycle, s5_fog, s6_compound]
