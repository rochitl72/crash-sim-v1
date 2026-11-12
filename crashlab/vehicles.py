"""Vehicle catalogue for CrashLab v1.

All numbers are *typical class values* (rounded, from public spec sheets and
textbook ranges), NOT measured data for a specific make/model. For a real
case replace them with the actual vehicle's mass, dimensions and CG location.

This is the original, physics-only catalogue: just enough per vehicle to
drive the rigid-body dynamics and SAT collision model. There is no crush/
damage model, no airbag/occupant data and no roll/tip-over geometry here —
those were added later in v2.
"""
import numpy as np

G = 9.81


class Spec:
    """Rigid-body vehicle description. Body frame: x forward, y left, origin at CG."""

    def __init__(self, name, kind, mass, izz, length, width, a, b, h, wheels,
                 c_alpha=9.0, cda=0.7, color="#888888", slide_on_hit=False,
                 max_steer=0.55, drive_max_g=0.30, crr=0.015, slide_mu=0.5):
        self.name, self.kind = name, kind
        self.mass, self.izz = float(mass), float(izz)
        self.length, self.width = float(length), float(width)
        self.a, self.b, self.h = float(a), float(b), float(h)   # a: CG->front axle, b: CG->rear axle, h: CG height
        self.wheels = np.asarray(wheels, float)   # x, y, steer, drive, brake, fz0
        self.c_alpha, self.cda, self.color = c_alpha, cda, color
        self.slide_on_hit, self.max_steer = slide_on_hit, max_steer
        self.drive_max_g, self.crr, self.slide_mu = drive_max_g, crr, slide_mu
        w = self.wheels
        front = w[:, 0] > 0
        self.klong = np.where(front, -1.0 / max(front.sum(), 1), 1.0 / max((~front).sum(), 1))
        yy = w[:, 1]
        self.klat = np.where(np.abs(yy) > 1e-6,
                             -np.sign(yy) * 0.5 / (2 * np.maximum(np.abs(yy), 1e-6)), 0.0)
        # mild understeer bias (real cars are understeer-balanced; equal normalised
        # stiffness would be exactly neutral-steer). Not applied to bikes / 3-wheelers.
        us = 0.12 if (front.sum() == 2 and (~front).sum() == 2) else 0.0
        self.ca_mult = np.where(front, 1.0 - us, 1.0 + us)
        self.rect_off = (self.a - self.b) / 2.0          # geometric centre ahead of CG
        self.radius = 0.5 * np.hypot(self.length, self.width)
        self.wheelbase = self.a + self.b

    def meta(self):
        return dict(name=self.name, kind=self.kind, mass=self.mass, length=self.length,
                    width=self.width, rect_off=self.rect_off, color=self.color, cg_h=self.h,
                    a=self.a, b=self.b,
                    wheels=[[float(x), float(y), int(s)] for x, y, s, *_ in self.wheels])


def _make(name, kind, mass, izz, L, W, a, b, h, nf, nr, tf, tr, drive, brake_front,
          c_alpha, cda, color, **kw):
    ff, fr = b / (a + b), a / (a + b)
    wl = []
    for i in range(nf):
        y = 0.0 if nf == 1 else (tf / 2 if i == 0 else -tf / 2)
        wl.append([a, y, 1.0, (1.0 / nf if drive == "front" else 0.0), brake_front / nf, ff / nf])
    for i in range(nr):
        y = 0.0 if nr == 1 else (tr / 2 if i == 0 else -tr / 2)
        wl.append([-b, y, 0.0, (1.0 / nr if drive == "rear" else 0.0), (1 - brake_front) / nr, fr / nr])
    return Spec(name, kind, mass, izz, L, W, a, b, h, wl, c_alpha, cda, color, **kw)


def sedan(color="#2ec4b6"):
    return _make("Sedan", "car", 1250, 1900, 4.50, 1.80, 1.25, 1.40, 0.55, 2, 2, 1.55, 1.55,
                 "front", 0.70, 9.5, 0.70, color)


def hatchback(color="#ffb703"):
    return _make("Hatchback", "car", 1000, 1350, 3.85, 1.70, 1.05, 1.40, 0.55, 2, 2, 1.45, 1.45,
                 "front", 0.70, 9.0, 0.65, color)


def suv(color="#8ecae6"):
    return _make("SUV", "car", 1900, 3300, 4.70, 1.85, 1.30, 1.45, 0.72, 2, 2, 1.55, 1.55,
                 "rear", 0.68, 8.5, 1.00, color)


def truck(color="#e76f51"):
    return _make("Goods truck (laden)", "truck", 9000, 52000, 8.00, 2.50, 3.00, 1.80, 1.10, 2, 2,
                 2.00, 1.85, "rear", 0.55, 7.0, 5.00, color)


def motorcycle(color="#b5179e"):
    return _make("Motorcycle + rider", "two-wheeler", 260, 55, 2.05, 0.75, 0.70, 0.65, 0.65, 1, 1,
                 0.0, 0.0, "rear", 0.60, 7.0, 0.50, color, slide_on_hit=True, max_steer=0.5, slide_mu=0.45)


CATALOGUE = [sedan, hatchback, suv, truck, motorcycle]
