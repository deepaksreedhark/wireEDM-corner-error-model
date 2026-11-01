# -*- coding: utf-8 -*-
"""
simplified_corner_gui_v11.py — corner-error model for wire EDM rough cuts (single file)
=====================================================================================

Version v11 (1 Nov 2026). One file holds the calculation, the plots, the browser view and the window.
Keep plotly.min.js in the same folder so the browser view works offline.

Model (two steps)
-----------------
1. Mean wire lag: tensioned string between the guides, spark force F spread over the cut
       delta_mid = F (2L - H) / 8T     (Obara 2003 eq. 2)    delta_face = F (L - H) / 4T   (Chen 2014 eq. 4)
2. Corner pursuit: the mid-height wire trails the guide by delta_mid along its own travel
   (Han 2007 eqs. 14-15). The whole wire moves as one body (step model, Step 11), so at height z
   the offset from the guide is y(z)/delta_mid times the mid-height offset. Corner path error =
   closest approach to the programmed corner C (Chen 2014), reported at five heights.
3. (v6, optional) Discharge-circle force: instead of assuming the push always points straight back,
   the spark force is taken from the part of the discharge circle (radius r_d) that faces the
   direction of travel and is still uncut: w = max(0, n.v)[uncut], f = -sum w n / sum_straight.
   The wire sits where the string balances it, W = G + delta_mid * f(W). Options: discharge
   concentration c (S_rate = S_area^(1-c)) and gap growth dg = k_g (m - 1). Outputs: spark-rate
   dip, force size and rotation, edge error at the part tip (Wang 2022), and the machined part
   coloured by the material removed beyond the ideal edge.
4. (v7, optional) Process force law instead of a fixed F, fitted to the 27-run L27 experiment of
   Chen 2014 (Table 4) through the pursuit model:
       F = K (v_f H/100)^1.456 (t_on/12)^1.258 (t_off/12)^-0.203 (V_p/40)^0.746 (T/10)^0.843
   (v9: removal-rate form; v7-v8 used (v_f/4)^1.456, identical at Chen's H = 25 mm)
   v_f feed [mm/min], H thickness [mm], t_on / t_off pulse on / off [µs], V_p peak voltage [V], T tension [N].
   K is the force at the reference settings (v_f H = 100 mm²/min, 12/12 µs, 40 V, 10 N); 0.0701 N for Chen's
   machine. Set K for another machine from one measured straight-cut lag (button in the window).
   The tension exponent is empirical (see the methodology document, Section 7).
v8: the discharge model's uncut test is a smooth 3 µm contact zone centred on the edge of the old slot
   (U = clip(0.5 + d/eps, 0, 1)) instead of on/off, and the balance is solved with damping 0.2 and up to
   150 iterations. This removes the erratic path below 90°, where the on/off test stopped the solver converging.

Window: inputs, results table, a guide-position slider and matplotlib plots (3-D view, lag through the
height, top view). The 3-D view keeps its rotation (and zoom, while only the guide moves) when it
redraws. The browser button writes corner_view_v11.html (Plotly): machined part as a closed
triangle mesh (walls lofted between cross-sections, ear-clipped caps), slot, scrap side toggle, wire slider.

Run:  python simplified_corner_gui_v11.py
Needs: numpy, matplotlib, tkinter (python.org Windows installer).
"""

from __future__ import annotations

import sys


# =============================================================================== MODEL
import math
from dataclasses import dataclass, field

import numpy as np

VERSION = "v11 (1 Nov 2026)"
APP_TITLE = "Wire-EDM corner error modelling v11 1 Nov 2026"

DEFAULT_F_N = 0.037   # total spark force [N]: Chen 2014 baseline (his measured 30.7 µm lag)

# Process force law, fitted to Chen 2014 Table 4 (27 runs, corner error at 90°, pursuit model):
#   F = K * prod (x / x_ref)^exponent      leave-one-out error 13 % (48 % without the tension term)
# v9: the feed term is the removal rate v_f*H [mm²/min] (reference 4 mm/min x 25 mm = 100), so thickness
# acts like feed. On Chen's 25 mm data this is identical to v7's (v_f/4); it fixes the thickness trend of
# Obara 2003 and Liang & Liao 2014, where thicker parts were cut at lower feed.
FORCE_LAW_K = 0.0701      # [N] force at the reference settings on Chen's machine
FORCE_LAW = (             # input name, reference value, exponent, label
    ("removal_rate", 100.0, 1.456, "removal rate v_f·H"),
    ("t_on_us", 12.0, 1.258, "pulse-on t_on"),
    ("t_off_us", 12.0, -0.203, "pulse-off t_off"),
    ("V_peak", 40.0, 0.746, "peak voltage V_p"),
    ("T_N", 10.0, 0.843, "tension T (empirical)"),
)

SECTIONS = [          # name, height as a fraction of H from the bottom face
    ("Top face", 1.00),
    ("Upper quarter", 0.75),
    ("Mid-height", 0.50),
    ("Lower quarter", 0.25),
    ("Bottom face", 0.00),
]


@dataclass
class Inputs:
    angle_deg: float = 90.0     # part corner angle theta [deg]
    H_mm: float = 25.0          # workpiece thickness [mm]
    c_mm: float = 10.0          # clearance from each workpiece face to its guide [mm]
    T_N: float = 9.8            # wire tension [N]
    F_N: float = DEFAULT_F_N    # total spark force [N]
    d_mm: float = 0.25          # wire diameter [mm] (only for drawing the part)
    gap_um: float = 25.0        # spark gap [µm] (discharge radius r_d = d/2 + g)
    corner_model: str = "pursuit"   # "pursuit" (v5) or "discharge" (force from the discharge circle)
    conc: float = 0.0           # discharge concentration c (0 = sparks fall with the uncut arc, 1 = same spark count)
    k_g_um: float = 0.0         # gap growth per unit rise in sparks per unit material [µm] (0 = off)
    force_mode: str = "fixed"   # "fixed" (F_N) or "law" (v7 process force law)
    feed_mm_min: float = 4.0    # feed rate [mm/min]
    t_on_us: float = 12.0       # pulse-on time [µs]
    t_off_us: float = 12.0      # pulse-off time [µs]
    V_peak: float = 40.0        # peak (gap) voltage [V]
    K_N: float = FORCE_LAW_K    # machine constant: force at the reference settings [N]

    @property
    def removal_rate(self):
        """Feed times thickness [mm²/min], proportional to the metal removed per second."""
        return self.feed_mm_min * self.H_mm

    @property
    def F_law_factor(self):
        """Product of (x / x_ref)^exponent over the force-law inputs."""
        return float(np.prod([(getattr(self, k) / ref) ** e for k, ref, e, _ in FORCE_LAW]))

    @property
    def F_eff(self):
        """Spark force used by the model [N]."""
        return self.K_N * self.F_law_factor if self.force_mode == "law" else self.F_N

    @property
    def H(self): return self.H_mm * 1e-3

    @property
    def c(self): return self.c_mm * 1e-3

    @property
    def L(self): return self.H + 2.0 * self.c

    @property
    def r_d(self): return 0.5 * self.d_mm * 1e-3 + self.gap_um * 1e-6

    def check(self):
        if not 0.0 < self.angle_deg < 180.0:
            raise ValueError("Corner angle must be between 0 and 180 deg.")
        for name, label in (("H_mm", "Thickness"), ("T_N", "Tension"), ("F_N", "Force"), ("d_mm", "Wire diameter")):
            if getattr(self, name) <= 0:
                raise ValueError(f"{label} must be positive.")
        if self.c_mm < 0 or self.gap_um < 0:
            raise ValueError("Clearance and gap cannot be negative.")
        if self.corner_model not in ("pursuit", "discharge"):
            raise ValueError("Corner model must be 'pursuit' or 'discharge'.")
        if not 0.0 <= self.conc <= 1.0:
            raise ValueError("Discharge concentration c must be between 0 and 1.")
        if self.k_g_um < 0:
            raise ValueError("Gap growth k_g cannot be negative.")
        if self.force_mode not in ("fixed", "law"):
            raise ValueError("Force mode must be 'fixed' or 'law'.")
        if self.force_mode == "law":
            for name, label in (("feed_mm_min", "Feed"), ("t_on_us", "Pulse-on time"), ("t_off_us", "Pulse-off time"),
                                ("V_peak", "Peak voltage"), ("K_N", "Machine constant K")):
                if getattr(self, name) <= 0:
                    raise ValueError(f"{label} must be positive.")


def K_from_lag(p: "Inputs", lag_um: float) -> float:
    """Machine constant K that makes the straight-cut mid-height lag equal lag_um at the current settings."""
    if lag_um <= 0:
        raise ValueError("Measured lag must be positive.")
    per_newton = (2 * p.L - p.H) / (8 * p.T_N)            # delta_mid per unit force [m/N]
    return lag_um * 1e-6 / per_newton / p.F_law_factor


# ----------------------------------------------------------------------------- step 1: lag
def lag(p: Inputs, z):
    """Wire lag y(z) behind the guide line [m]; z from the lower guide [m]."""
    z = np.asarray(z, dtype=float)
    F, T, H, c, L = p.F_eff, p.T_N, p.H, p.c, p.L
    return np.where(z < c, F / (2 * T) * z,
                    np.where(z <= c + H, F / (2 * T) * z - F / (2 * T * H) * (z - c) ** 2,
                             F / (2 * T) * (L - z)))


# ----------------------------------------------------------------------------- step 2: pursuit
def leg_directions(angle_deg):
    turn = math.radians(180.0 - angle_deg)
    d1 = np.array([1.0, 0.0])
    d2 = np.array([math.cos(-turn), math.sin(-turn)])
    return d1, d2


def guide_path(angle_deg, lead, step):
    """Programmed path: leg 1 along +x into C = (0, 0), then leg 2. Also returns arc length s (0 at C)."""
    d1, d2 = leg_directions(angle_deg)
    s1 = np.arange(-lead, 0.0, step)
    s2 = np.arange(0.0, lead + 0.5 * step, step)
    return np.vstack([np.outer(s1, d1), np.outer(s2, d2)]), np.concatenate([s1, s2])


def pursue(guide, delta, start_dir):
    """Wire point trailing the guide by delta along its own travel."""
    W = np.empty_like(guide)
    w = guide[0] - delta * start_dir
    for k, g in enumerate(guide):
        e = g - w
        n = math.hypot(e[0], e[1])
        if n > 0.0:
            w = g - delta * e / n
        W[k] = w
    return W


def part_tip(p: Inputs):
    """Ideal part vertex P: the part edges are the legs offset by r_d to the right of travel."""
    d1, d2 = leg_directions(p.angle_deg)
    n1, n2 = np.array([d1[1], -d1[0]]), np.array([d2[1], -d2[0]])
    return p.r_d * (n1 + n2) / (1.0 + n1 @ n2)


# ----------------------------------------------------------------------------- step 2b: discharge-circle force (v6)
def discharge_corner(angle_deg, delta0, r_d, c=0.0, k_g=0.0, step=None, n_circle=180,
                     before=None, after=None, iters=150, damping=0.2, v_window=None, rigid=False, v_mode="pursuit", m_max=5.0, ex_frac=0.2,
                     contact=3e-6):
    """Quasi-static force balance of the mid-height wire round a sharp corner.

    At each guide position G_k the wire sits at W_k = G_k + delta0 * f(W_k), where f is the spark
    force (per unit straight-cut force) from the points of the discharge circle that face the wire's
    direction of travel v and are still uncut (outside every earlier discharge circle).
        w(n) = max(0, n.v) [uncut],   S_area = sum w dphi / sum_straight,   f_area = -sum w n dphi / sum_straight
        S_rate = S_area^(1-c),   f = f_area * S_rate / S_area,   m = S_rate / S_area,   dg = k_g (m - 1)
    rigid=True keeps the wire on the guide (delta0 is ignored for the position).
    Returns a dict of per-step arrays.
    """
    d1, d2 = leg_directions(angle_deg)
    step = step or (min(delta0 / 20.0, r_d / 100.0) if delta0 > 0 else r_d / 100.0)
    before = before if before is not None else 2.5 * r_d + 4 * delta0
    half = math.radians(180.0 - angle_deg) / 2
    cot = 1.0 / math.tan(max(math.radians(angle_deg) / 2, 1e-3))
    after = after if after is not None else 1.3 * r_d * cot + 2.5 * r_d + 8 * delta0
    s = np.arange(-before, after + 0.5 * step, step)
    G = np.where(s[:, None] < 0, s[:, None] * d1[None, :], s[:, None] * d2[None, :])

    phi = (np.arange(n_circle) + 0.5) * 2 * math.pi / n_circle
    N = np.column_stack([np.cos(phi), np.sin(phi)])          # outward normals of the circle
    dphi = 2 * math.pi / n_circle
    # numerical straight-cut reference (removes quadrature error)
    w_ref = np.maximum(0.0, N @ d1)
    S_ref = w_ref.sum() * dphi
    F_ref = -(w_ref[:, None] * N).sum(axis=0) * dphi        # = (-pi/2, 0) analytically
    F_ref_len = float(np.hypot(*F_ref))

    # direction of travel from the wire's previous positions over a baseline of about 1.5 delta0:
    # a shorter baseline makes the force direction react faster than the lag can follow (unstable)
    v_window = v_window or (max(2, int(round(1.5 * delta0 / step))) if delta0 > 0 else 2)

    # history: steady straight trail before the first computed point (fills the slot behind)
    n_pre = int(math.ceil((2.2 * r_d + 2 * delta0) / step))
    s_pre = s[0] - step * np.arange(n_pre, 0, -1)
    lag0 = 0.0 if rigid else delta0
    H_W = list(s_pre[:, None] * d1[None, :] - lag0 * d1[None, :])
    H_r = [r_d] * n_pre

    # the last few micrometres of the wire's own track are the continuous cutting front, not old slot:
    # leave them out of the uncut test (otherwise a wire that lags by a hair sees no material at all)
    n_ex = 1 if rigid else max(2, int(math.ceil(ex_frac * max(delta0, 1e-6) / step)))
    out = {k: [] for k in ("W", "f", "S_area", "S_rate", "m", "r", "v")}
    W = G[0] - lag0 * d1
    for k in range(len(s)):
        g = G[k]
        histW_all = np.asarray(H_W)
        histR_all = np.asarray(H_r)
        histW, histR = histW_all[:-n_ex], histR_all[:-n_ex]
        # direction of travel of the wire from its recent motion
        # direction of travel: from the previous wire position towards the guide (pursuit line,
        # Han et al. 2007); this is what the spark weights are taken against
        if v_mode == "pursuit":
            vv = g - histW_all[-1]
        else:                                    # actual wire motion over ~1.5 delta0
            vv = histW_all[-1] - histW_all[max(0, len(H_W) - 1 - v_window)]
        v = vv / max(np.hypot(*vv), 1e-15)
        if rigid:
            v = (g - G[k - 1]) / step if k else d1
        guess = W + (g - (G[k - 1] if k else g - step * d1))
        for _ in range(iters):
            # candidate history circles that can cover this circle
            r_cur = histR_all[-1]
            dc = np.hypot(*(histW - guess).T)
            near = dc < (histR + r_cur + 0.5 * contact + 1e-9)
            P = guess[None, :] + r_cur * N                     # circle points
            if near.any():
                hW, hR = histW[near], histR[near]
                d = np.sqrt(((P[:, None, :] - hW[None, :, :]) ** 2).sum(-1)) - hR[None, :]
                dmin = d.min(axis=1)                          # clearance outside the old slot [m]
                if rigid or contact <= 0:
                    uncut = (dmin > 1e-9).astype(float)
                else:                                        # smooth contact zone centred on the slot edge (v8)
                    uncut = np.clip(0.5 + dmin / contact, 0.0, 1.0)
            else:
                uncut = np.ones(len(P))
            w = np.maximum(0.0, N @ v) * uncut
            S_area = w.sum() * dphi / S_ref
            f_area = -(w[:, None] * N).sum(axis=0) * dphi / F_ref_len
            if S_area > 1e-12:
                m = min(S_area ** (-c), m_max)           # sparks per unit material, capped
                S_rate = m * S_area
            else:
                m, S_rate = 1.0, 0.0
            f = f_area * m
            target = g if rigid else g + delta0 * f
            new = (1 - damping) * guess + damping * target
            if np.hypot(*(new - guess)) < 1e-9:
                guess = new
                break
            guess = new
        W = guess
        r_new = r_d + k_g * (m - 1.0)
        H_W.append(W.copy()); H_r.append(r_new)
        for key, val in (("W", W.copy()), ("f", f), ("S_area", S_area), ("S_rate", S_rate), ("m", m), ("r", r_new), ("v", v)):
            out[key].append(val)
    res = {k: np.asarray(v) for k, v in out.items()}
    res.update(s=s, G=G, step=step)
    dist = np.hypot(*res["W"].T)
    i = int(dist.argmin())
    res.update(error=float(dist[i]), closest=res["W"][i], i_closest=i)
    return res


# ----------------------------------------------------------------------------- full run
@dataclass
class Result:
    inputs: Inputs
    delta_mid: float
    delta_face: float
    sections: list               # dicts: name, z, lag, path (N x 2), error, closest
    guide: np.ndarray
    s: np.ndarray                # arc length along the guide path, 0 at C [m]
    wire_mid: np.ndarray
    box: tuple                   # plotted region (lo, hi) round C, square [m]
    tip: np.ndarray = field(default_factory=lambda: np.zeros(2))
    radius: np.ndarray | None = None   # discharge radius at each path point [m] (gap growth)
    spark: dict | None = None          # discharge model: f, S_area, S_rate, m, v per path point
    e_pursuit: float = 0.0             # mid-height corner error of the pursuit model [m], for comparison

    def section(self, name):
        return next(sec for sec in self.sections if sec["name"] == name)


def run(p: Inputs, resolution: int = 1000) -> Result:
    p.check()
    zm = p.c + 0.5 * p.H
    d_mid = float(lag(p, zm))
    d_face = float(lag(p, p.c))
    step = d_mid / resolution
    guide, s = guide_path(p.angle_deg, lead=6.0 * d_mid, step=step)
    d1, _ = leg_directions(p.angle_deg)
    wire_mid = pursue(guide, d_mid, d1)
    e_pursuit = float(np.hypot(*wire_mid.T).min())
    radius, spark = np.full(len(s), p.r_d), None
    if p.corner_model == "discharge":
        dc = discharge_corner(p.angle_deg, d_mid, p.r_d, c=p.conc, k_g=p.k_g_um * 1e-6)
        guide, s, wire_mid, radius = dc["G"], dc["s"], dc["W"], dc["r"]
        spark = {k: dc[k] for k in ("f", "S_area", "S_rate", "m", "v")}
    D = wire_mid - guide                                   # mid-height displacement from the guide
    secs = []
    for name, frac in SECTIONS:
        z = p.c + frac * p.H
        dz = float(lag(p, z))
        path = guide + (dz / d_mid) * D
        dist = np.hypot(path[:, 0], path[:, 1])
        i = int(dist.argmin())
        secs.append(dict(name=name, frac=frac, z=z, lag=dz, path=path, error=float(dist[i]), closest=path[i]))
    # square plotting window round the corner: everything within 2.5 lag lengths of C
    pts = np.vstack([guide, wire_mid])
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < 2.5 * d_mid]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    half, mid = 0.5 * max(hi - lo) + 0.7 * d_mid, 0.5 * (lo + hi)
    res = Result(p, d_mid, d_face, secs, guide, s, wire_mid, (mid - half, mid + half), part_tip(p),
                 radius=radius, spark=spark, e_pursuit=e_pursuit)
    for sec in res.sections:
        sec["edge"] = edge_error(res, sec)
    return res


def edge_error(r: Result, sec) -> float:
    """Wang 2022 edge error at one height: material removed from the ideal part tip P, measured
    along the corner bisector into the part [m]. 0 = tip intact. The slot at this height is the
    envelope of discharge circles of radius r_k round the wire path."""
    d1, d2 = leg_directions(r.inputs.angle_deg)
    b = (d2 - d1) / np.linalg.norm(d2 - d1)                         # from C towards P, into the part
    P = r.tip
    path, rad = sec["path"], r.radius
    near = np.hypot(*(path - P).T) < rad.max() + 3 * r.delta_mid + 50e-6
    if not near.any():
        return 0.0
    pp, rr = path[near], rad[near]
    t = np.arange(0.0, 3 * r.delta_mid + 30e-6, 0.05e-6)
    pts = P[None, :] + t[:, None] * b[None, :]
    clear = (np.sqrt(((pts[:, None, :] - pp[None, :, :]) ** 2).sum(-1)) - rr[None, :]).min(axis=1)
    keep = np.nonzero(clear > -0.02e-6)[0]                          # tangency at the ideal tip counts as kept
    return float(t[keep[0]]) if len(keep) else float(t[-1])


def summary(r: Result):
    """Rows for a results table: (quantity, value)."""
    p = r.inputs
    model_name = "discharge-circle force" if p.corner_model == "discharge" else "pursuit"
    rows = [("Corner model", model_name),
            ("Spark force F" + (" (process law)" if p.force_mode == "law" else " (fixed)"), f"{p.F_eff * 1e3:.1f} mN"),
            *((("  removal rate v_f·H", f"{p.removal_rate:g} mm²/min"),) if p.force_mode == "law" else ()),
            ("Guide span L = H + 2c", f"{p.L * 1e3:.1f} mm"),
            ("Mid-height lag δ_mid = F(2L − H)/8T", f"{r.delta_mid * 1e6:.1f} µm"),
            ("Face lag δ_face = F(L − H)/4T", f"{r.delta_face * 1e6:.1f} µm"),
            (f"κ({p.angle_deg:g}°) = e_mid / δ_mid", f"{r.section('Mid-height')['error'] / r.delta_mid:.3f}")]
    if r.spark is not None:
        sp = r.spark
        fmag = np.hypot(*sp["f"].T)
        rot = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", -sp["f"] / np.maximum(fmag, 1e-12)[:, None], sp["v"]), -1, 1)))
        rows += [("  pursuit e_mid, for comparison", f"{r.e_pursuit * 1e6:.1f} µm"),
                 ("  min spark rate S/S₀", f"{sp['S_rate'].min():.3f}"),
                 ("  min force |f| (per unit straight-cut force)", f"{fmag.min():.3f}"),
                 ("  max force rotation from travel direction", f"{rot.max():.1f}°"),
                 ("  max sparks per unit material m", f"{sp['m'].max():.2f}")]
        if p.k_g_um > 0:
            rows.append(("  max gap growth Δg", f"{(r.radius.max() - p.r_d) * 1e6:.1f} µm"))
    rows.append(("Section (height from lower guide)", "lag δ | error e | edge"))
    for sec in r.sections:
        rows.append((f"  {sec['name']}, z = {sec['z'] * 1e3:.1f} mm",
                     f"{sec['lag'] * 1e6:.1f} | {sec['error'] * 1e6:.1f} | {sec['edge'] * 1e6:.1f} µm"))
    return rows


# ----------------------------------------------------------------------------- plots
SERIF = ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"]


def set_style(base_pt: float = 10.0) -> None:
    """Times New Roman for all plot text; every label is sized relative to base_pt."""
    import logging
    import matplotlib as mpl
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # quiet if a font is missing
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": SERIF, "mathtext.fontset": "stix",
        "font.size": base_pt, "axes.titlesize": "medium", "axes.labelsize": "medium",
        "xtick.labelsize": "small", "ytick.labelsize": "small", "legend.fontsize": "small",
    })


SECTION_STYLE = {   # colour, line style; faces and quarters are symmetric, so the lower ones are dashed
    "Top face": ("#d62728", "-"),
    "Upper quarter": ("#ff7f0e", "-"),
    "Mid-height": ("#1f77b4", "-"),
    "Lower quarter": ("#ff7f0e", "--"),
    "Bottom face": ("#d62728", "--"),
}
WIRE_COLOUR = "#2ca02c"
SHORT = {"Top face": "top", "Upper quarter": "upper ¼", "Mid-height": "mid", "Lower quarter": "lower ¼", "Bottom face": "bottom"}


def _inside(path, box):
    lo, hi = box
    m = np.all((path >= lo) & (path <= hi), axis=1)
    return path[m]


def _part_arrow(r: Result):
    """From C towards the part tip P, stopping inside the window [m]."""
    lo, hi = r.box
    tip = r.tip
    b = tip / np.linalg.norm(tip)
    span = min(hi - lo)
    length = min(float(np.linalg.norm(tip)), 0.42 * span)
    return b * length, float(np.linalg.norm(tip))


def snapshot(r: Result, s_guide: float):
    """Guide position and the 3-D wire shape when the guide is s_guide [m] along the path from C."""
    k = int(np.clip(np.searchsorted(r.s, s_guide), 0, len(r.s) - 1))
    G = r.guide[k]
    D = r.wire_mid[k] - G
    z = np.linspace(0.0, r.inputs.L, 121)
    wire = G[None, :] + (lag(r.inputs, z) / r.delta_mid)[:, None] * D[None, :]
    return G, wire, z


def plot_3d(r: Result, ax, s_guide: float | None = None):
    """Workpiece slab, guides, deflected wire and the corner paths at five heights.

    x, y in µm (cutting plane, zoomed on the corner); height in mm. The scales differ on
    purpose: the lag is micrometres, the height tens of millimetres.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    p, um, mm = r.inputs, 1e6, 1e3
    (x0, y0), (x1, y1) = r.box[0] * um, r.box[1] * um
    s_guide = 1.5 * r.delta_mid if s_guide is None else s_guide
    zb, zt = p.c * mm, (p.c + p.H) * mm

    # workpiece slab (thickness H) over the plotted window
    rect = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    faces = [[(x, y, zb) for x, y in rect], [(x, y, zt) for x, y in rect]]
    ax.add_collection3d(Poly3DCollection(faces, facecolor="0.62", edgecolor="0.4", alpha=0.4, linewidths=0.8))
    for x, y in rect:
        ax.plot([x, x], [y, y], [zb, zt], color="0.6", lw=0.6)
    ax.text(x0, y1, zt, "  top face", fontsize="small", color="0.35")
    ax.text(x0, y1, zb, "  bottom face", fontsize="small", color="0.35")

    # programmed path on both faces, corner line through C, arrow to the part
    g = _inside(r.guide, r.box) * um
    for zz in (zb, zt):
        ax.plot(g[:, 0], g[:, 1], zz, color="k", lw=0.9, alpha=0.7)
    ax.plot([0, 0], [0, 0], [zb, zt], color="k", lw=0.8, ls=":")
    ax.text(0, 0, zt + 0.04 * p.L * mm, "C", fontsize="medium", fontweight="bold")
    a, dist = _part_arrow(r)
    a = a * um
    ax.plot([0, a[0]], [0, a[1]], [zt, zt], color="0.3", lw=1.2)
    ax.text(a[0], a[1], zt, f" part side\n (tip P {dist * um:.0f} µm)", fontsize="small", color="0.3")

    # wire paths at the five sections
    for sec in r.sections:
        col, ls = SECTION_STYLE[sec["name"]]
        path = _inside(sec["path"], r.box) * um
        ax.plot(path[:, 0], path[:, 1], sec["z"] * mm, color=col, ls=ls, lw=1.8)

    # wire and guides at the chosen moment
    G, wire, zw = snapshot(r, s_guide)
    Gu = G * um
    ax.plot([Gu[0]] * 2, [Gu[1]] * 2, [0, p.L * mm], color="0.45", lw=0.8, ls="--")
    ax.plot(wire[:, 0] * um, wire[:, 1] * um, zw * mm, color=WIRE_COLOUR, lw=2.6)
    for zz, name in ((0.0, "lower guide"), (p.L * mm, "upper guide")):
        ax.scatter([Gu[0]], [Gu[1]], [zz], marker="s", s=60, color="0.2", depthshade=False)
        ax.text(Gu[0], Gu[1], zz, f"   {name}", fontsize="small")
    for sec in r.sections:
        col, _ = SECTION_STYLE[sec["name"]]
        k = int(np.argmin(np.abs(zw - sec["z"])))
        ax.scatter([wire[k, 0] * um], [wire[k, 1] * um], [zw[k] * mm], color=col, s=20, depthshade=False)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(0, p.L * mm)
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_zlabel("height from lower guide [mm]")
    ax.set_box_aspect((1, 1, 1.3))
    ax.view_init(elev=22, azim=-60)
    ax.set_title("3-D view: wire (green) between the guides; coloured lines = wire path at each height\n"
                 "x, y in µm, zoomed on the corner; height in mm (drag to rotate)", fontsize="medium")


def plot_plan(r: Result, ax):
    """Top view of the corner: programmed path and the wire path at the five heights."""
    um = 1e6
    (x0, y0), (x1, y1) = r.box[0] * um, r.box[1] * um
    g = _inside(r.guide, r.box) * um
    ax.plot(g[:, 0], g[:, 1], color="k", lw=1.1, label="programmed path")
    ax.plot([0], [0], "ko", ms=4)
    ax.annotate("C", (0, 0), xytext=(4, 4), textcoords="offset points", fontweight="bold")
    a, dist = _part_arrow(r)
    a = a * um
    ax.annotate("", xy=(a[0], a[1]), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="0.35"))
    ax.annotate(f"part side (tip P {dist * um:.0f} µm)", (a[0], a[1]), xytext=(0, -10),
                textcoords="offset points", fontsize="small", color="0.35", ha="center")
    for sec in r.sections:
        col, ls = SECTION_STYLE[sec["name"]]
        path = _inside(sec["path"], r.box) * um
        ax.plot(path[:, 0], path[:, 1], color=col, ls=ls, lw=1.6,
                label=f"{SHORT[sec['name']]}: {sec['error'] * 1e6:.1f} µm")
        cp = sec["closest"] * um
        ax.plot([0, cp[0]], [0, cp[1]], color=col, lw=0.8, ls=ls)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_title("Top view (looking down the wire)", fontsize="medium")
    ax.legend(fontsize="x-small", loc="lower left", framealpha=0.75, title="e at", title_fontsize="x-small",
              handlelength=1.5, borderpad=0.3, labelspacing=0.2)


def plot_profile(r: Result, ax):
    """Lag through the height, with the five sections marked."""
    p, um, mm = r.inputs, 1e6, 1e3
    z = np.linspace(0, p.L, 301)
    ax.axhspan(p.c * mm, (p.c + p.H) * mm, color="0.9", label="workpiece")
    ax.plot(lag(p, z) * um, z * mm, color=WIRE_COLOUR, lw=2, label="wire lag y(z)")
    ax.axvline(0, color="0.5", lw=0.8, ls="--")
    for sec in r.sections:
        col, _ = SECTION_STYLE[sec["name"]]
        ax.plot([sec["lag"] * um], [sec["z"] * mm], "o", color=col)
        ax.annotate(f"{sec['lag'] * um:.1f} µm", (sec["lag"] * um, sec["z"] * mm),
                    xytext=(5, -3), textcoords="offset points", fontsize="small")
    ax.set_xlim(-0.05 * r.delta_mid * um, 1.45 * r.delta_mid * um)
    ax.set_ylim(0, p.L * mm)
    ax.set_xlabel("lag behind the guide [µm]")
    ax.set_ylabel("height from lower guide [mm]")
    ax.set_title("Lag through the height (side view)", fontsize="medium")
    ax.legend(fontsize="small", loc="lower right")


def plot_spark(r: Result, ax):
    """Discharge model: force size, force rotation and spark rate against guide position."""
    ax.set_title("Spark force and spark rate through the corner", fontsize="medium")
    if r.spark is None:
        ax.text(0.5, 0.5, "Pursuit model: force always 1.0, straight back.\nChoose 'Discharge-circle force'\nto see the spark force and spark rate.",
                ha="center", va="center", transform=ax.transAxes, color="0.35", fontsize="small")
        ax.set_xticks([]); ax.set_yticks([])
        return
    sp = r.spark
    x = r.s / r.delta_mid
    fmag = np.hypot(*sp["f"].T)
    rot = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", -sp["f"] / np.maximum(fmag, 1e-12)[:, None], sp["v"]), -1, 1)))
    ax.plot(x, fmag, color="#1f77b4", lw=1.8, label="force |f|")
    ax.plot(x, sp["S_rate"], color="#ff7f0e", lw=1.8, label="spark rate S/S₀")
    if r.inputs.conc > 0:
        ax.plot(x, sp["S_area"], color="#ff7f0e", lw=1.0, ls="--", label="uncut arc S_area")
    ax.axvline(0, color="0.5", lw=0.8, ls=":")
    ax.set_xlabel("guide position from C [× δ_mid]")
    ax.set_ylabel("per unit straight cut")
    lo = min(0.9, float(min(fmag.min(), sp["S_rate"].min())) - 0.05)
    ax.set_ylim(lo, max(1.05, float(sp["S_rate"].max()) + 0.05))
    ax.set_xlim(-3, min(12, x[-1]))
    ax2 = ax.twinx()
    rot_s = np.convolve(rot, np.ones(7) / 7, mode="same")             # smooth the 2° circle resolution
    ax2.plot(x, rot_s, color="#2ca02c", lw=1.2, ls="--", label="force rotation [deg]")
    ax2.set_ylabel("force rotation from travel [deg]", color="#2ca02c")
    ax2.tick_params(axis="y", colors="#2ca02c")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize="small", loc="lower right")


def get_view(fig, with_limits: bool = True) -> dict | None:
    """Current orientation (and axis limits) of the 3-D axes in fig, or None if there is none."""
    for ax in fig.axes:
        if getattr(ax, "name", "") == "3d":
            v = {"elev": ax.elev, "azim": ax.azim, "roll": getattr(ax, "roll", 0.0)}
            if with_limits:
                v["lims"] = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())
            return v
    return None


def plot_all(r: Result, fig=None, s_guide: float | None = None, view: dict | None = None):
    """Draw the three panels. view (from get_view) restores the 3-D orientation and, if it holds
    'lims', the 3-D zoom and pan."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    if "Times New Roman" not in mpl.rcParams["font.serif"][:1]:
        set_style(mpl.rcParams["font.size"])
    from matplotlib.gridspec import GridSpec
    if fig is None:
        fig = plt.figure(figsize=(13, 7.5))
    fig.clear()
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.45, 1], height_ratios=[1, 1.35, 1], wspace=0.25, hspace=0.55)
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    plot_3d(r, ax3, s_guide)
    if view:
        ax3.view_init(elev=view["elev"], azim=view["azim"], roll=view.get("roll", 0.0))
        if view.get("lims"):
            ax3.set_xlim(view["lims"][0])
            ax3.set_ylim(view["lims"][1])
            ax3.set_zlim(view["lims"][2])
    plot_profile(r, fig.add_subplot(gs[0, 1]))
    plot_plan(r, fig.add_subplot(gs[1, 1]))
    plot_spark(r, fig.add_subplot(gs[2, 1]))
    fig.subplots_adjust(left=0.0, right=0.93, top=0.95, bottom=0.07)
    return fig

model = sys.modules[__name__]          # the sections below refer to model.<name>


# =============================================================================== BROWSER VIEW
import json
import webbrowser
from pathlib import Path



CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


# ----------------------------------------------------------------------------- geometry fields
def _min_dist(points: np.ndarray, path: np.ndarray, chunk: int = 4000) -> np.ndarray:
    """Distance from each point (N x 2) to the nearest path point (M x 2)."""
    out = np.empty(len(points))
    for i in range(0, len(points), chunk):
        p = points[i:i + chunk]
        d2 = ((p[:, None, :] - path[None, :, :]) ** 2).sum(axis=-1)
        out[i:i + chunk] = np.sqrt(d2.min(axis=1))
    return out


def _min_clear(points: np.ndarray, path: np.ndarray, radius: np.ndarray, chunk: int = 2000) -> np.ndarray:
    """min over path points of (distance - discharge radius there): > 0 outside every discharge circle."""
    out = np.empty(len(points))
    for i in range(0, len(points), chunk):
        p = points[i:i + chunk]
        d = np.sqrt(((p[:, None, :] - path[None, :, :]) ** 2).sum(axis=-1)) - radius[None, :]
        out[i:i + chunk] = d.min(axis=1)
    return out


def _seg_dist(pts, a, b):
    """Distance from points (N x 2) to the segment a-b."""
    ab = b - a
    t = np.clip(((pts - a) @ ab) / max(float(ab @ ab), 1e-30), 0.0, 1.0)
    return np.hypot(*(pts - (a + np.outer(t, ab))).T)


def machined_fields(r: model.Result, half: float, nxy: int = 221, nz: int = 9):
    """Grid over the block and two fields (> 0 = material): part and scrap [m].

    Returns xs (grid line positions, same for x and y), zs (heights), part[ix, iy, k], scrap[ix, iy, k]."""
    p = r.inputs
    d1, d2 = model.leg_directions(p.angle_deg)
    n1, n2 = np.array([d1[1], -d1[0]]), np.array([d2[1], -d2[0]])
    P = r.tip
    xs = np.linspace(-half, half, nxy)
    zs = np.linspace(p.c, p.c + p.H, nz)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    inside = np.minimum((pts - P) @ n1, (pts - P) @ n2)          # > 0 inside the ideal part corner

    # wire offset from the guide at mid-height; at height z it is scaled by y(z)/delta_mid.
    # Beyond the computed stretch round C the wire trails steadily, so the slot there is bounded by
    # two straight segments (exact point-to-segment distance); the curved stretch is sampled at ~1 µm.
    D = r.wire_mid - r.guide
    reach = 1.5 * half + 2 * p.r_d
    step = max(1, int(round(1e-6 / (r.s[1] - r.s[0]))))
    Gc, Dc, Rc = r.guide[::step], D[::step], r.radius[::step]

    part = np.empty((nxy, nxy, nz))
    scrap = np.empty((nxy, nxy, nz))
    for k, z in enumerate(zs):
        f = float(model.lag(p, z)) / r.delta_mid
        curve = Gc + f * Dc
        a1, b1 = -reach * d1 + f * D[0], curve[0]                 # steady leg 1 up to the curve
        a2, b2 = curve[-1], reach * d2 + f * D[-1]                 # steady leg 2 after it
        slot = np.minimum(_min_clear(pts, curve, Rc),            # > 0 outside the slot
                          np.minimum(_seg_dist(pts, a1, b1), _seg_dist(pts, a2, b2)) - p.r_d)
        part[:, :, k] = np.minimum(slot, inside).reshape(nxy, nxy)
        scrap[:, :, k] = np.minimum(slot, -inside).reshape(nxy, nxy)
    return xs, zs, part, scrap


# ----------------------------------------------------------------------------- mesh of the machined block
def _outline(xs, F2):
    """Outer boundary of the largest region where F2 > 0 (F2[ix, iy] on the grid xs x xs).

    Marching squares (contourpy), so the boundary is interpolated between grid points. Returned
    counter-clockwise, without the repeated closing point."""
    import contourpy
    cg = contourpy.contour_generator(xs, xs, F2.T, fill_type=contourpy.FillType.OuterOffset)
    points, offsets = cg.filled(0.0, np.inf)
    best, best_area = None, 0.0
    for pts, off in zip(points, offsets):
        ring = pts[off[0]:off[1]]                       # first ring = outer boundary
        a = _area(ring)
        if abs(a) > best_area:
            best, best_area = ring, abs(a)
    if best is None:
        return np.zeros((0, 2))
    if np.allclose(best[0], best[-1]):
        best = best[:-1]
    return best if _area(best) > 0 else best[::-1]


def _area(ring):
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _resample(ring, n, anchor_dir):
    """n points evenly spaced along the closed ring, starting at the point furthest along anchor_dir
    (the same corner of the block at every height, so the rings line up for lofting)."""
    start = int(np.argmax(ring @ anchor_dir))
    ring = np.roll(ring, -start, axis=0)
    closed = np.vstack([ring, ring[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])])


def _simplify(ring, tol=2e-3):
    """Drop nearly collinear points (sine of the turn below tol); keeps corners and curves."""
    keep = []
    n = len(ring)
    for i in range(n):
        a, b, c = ring[i - 1], ring[i], ring[(i + 1) % n]
        u, v = b - a, c - b
        nu, nv = np.hypot(*u), np.hypot(*v)
        if nu == 0 or nv == 0:
            continue
        if abs(u[0] * v[1] - u[1] * v[0]) / (nu * nv) > tol:
            keep.append(i)
    return ring[keep] if len(keep) >= 3 else ring


def _earclip(P):
    """Triangulate a simple counter-clockwise polygon P (n x 2) by ear clipping. Returns (m x 3) indices."""
    idx = list(range(len(P)))
    tris = []

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    guard = 0
    while len(idx) > 3 and guard < 10 * len(P):
        guard += 1
        m = len(idx)
        clipped = False
        for t in range(m):
            i0, i1, i2 = idx[t - 1], idx[t], idx[(t + 1) % m]
            a, b, c = P[i0], P[i1], P[i2]
            if cross(a, b, c) <= 0:                       # reflex or flat: not an ear
                continue
            others = np.array([j for j in idx if j not in (i0, i1, i2)])
            if len(others):
                q = P[others]
                d1 = (b[0] - a[0]) * (q[:, 1] - a[1]) - (b[1] - a[1]) * (q[:, 0] - a[0])
                d2 = (c[0] - b[0]) * (q[:, 1] - b[1]) - (c[1] - b[1]) * (q[:, 0] - b[0])
                d3 = (a[0] - c[0]) * (q[:, 1] - c[1]) - (a[1] - c[1]) * (q[:, 0] - c[0])
                if np.any((d1 >= 0) & (d2 >= 0) & (d3 >= 0)):
                    continue                              # another vertex inside: not an ear
            tris.append((i0, i1, i2))
            idx.pop(t)
            clipped = True
            break
        if not clipped:                                   # numerical corner case: drop the flattest vertex
            m = len(idx)
            flat = min(range(m), key=lambda t: abs(cross(P[idx[t - 1]], P[idx[t]], P[idx[(t + 1) % m]])))
            idx.pop(flat)
    if len(idx) == 3:
        tris.append(tuple(idx))
    return np.array(tris, dtype=int)


def solid_mesh(xs, zs, F, anchor_dir, n_ring=480):
    """Closed triangle mesh of the region F > 0 (F[ix, iy, k]): side walls lofted between the
    cross-sections at heights zs, plus top and bottom caps. Returns vertices (V x 3) and faces (T x 3)."""
    rings = []
    for k in range(len(zs)):
        ring = _outline(xs, F[:, :, k])
        if len(ring) < 3:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
        rings.append(_resample(ring, n_ring, anchor_dir))
    nz, n = len(zs), n_ring
    verts = [np.column_stack([rg, np.full(n, z)]) for rg, z in zip(rings, zs)]
    faces = []
    j = np.arange(n)
    jn = (j + 1) % n
    for k in range(nz - 1):                               # walls: two triangles per quad
        a, b, c, d = k * n + j, k * n + jn, (k + 1) * n + jn, (k + 1) * n + j
        faces.append(np.column_stack([a, b, c]))
        faces.append(np.column_stack([a, c, d]))
    base = nz * n
    for k, flip in ((0, True), (nz - 1, False)):          # caps (bottom faces down, top faces up)
        cap = _simplify(rings[k])
        tri = _earclip(cap) + base
        if flip:
            tri = tri[:, ::-1]
        faces.append(tri)
        verts.append(np.column_stack([cap, np.full(len(cap), zs[k])]))
        base += len(cap)
    return np.vstack(verts), np.vstack(faces)


def _edge_deviation(r: model.Result, V: np.ndarray) -> np.ndarray:
    """For each part-mesh vertex: how far the machined surface lies inside the ideal part edge [m]
    (material removed beyond the ideal edge). Vertices far from the corner or on the plotted
    block's cut boundary get 0."""
    d1, d2 = model.leg_directions(r.inputs.angle_deg)
    n1, n2 = np.array([d1[1], -d1[0]]), np.array([d2[1], -d2[0]])
    rel = V[:, :2] - r.tip
    inside = np.minimum(rel @ n1, rel @ n2)
    limit = max(sec["edge"] for sec in r.sections) + 5e-6
    return np.where(inside < limit, np.maximum(inside, 0.0), 0.0)


# ----------------------------------------------------------------------------- figure
FONT_FAMILY = "Times New Roman, Times, serif"


def _wire_at(r: model.Result, s_guide: float):
    """3-D wire (121 points, lower to upper guide) and guide position for a guide s_guide [m]
    along the path from C. Beyond the computed path the wire trails steadily along the leg."""
    p = r.inputs
    d1, d2 = model.leg_directions(p.angle_deg)
    D = r.wire_mid - r.guide
    if s_guide < r.s[0]:
        G, Dk = s_guide * d1, D[0]
    elif s_guide > r.s[-1]:
        G, Dk = s_guide * d2, D[-1]
    else:
        k = int(np.clip(np.searchsorted(r.s, s_guide), 0, len(r.s) - 1))
        G, Dk = r.guide[k], D[k]
    z = np.linspace(0.0, p.L, 121)
    wire = G[None, :] + (model.lag(p, z) / r.delta_mid)[:, None] * Dk[None, :]
    return wire, G


def _r(a, nd=2):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def build_figure(r: model.Result, s_guide: float | None = None) -> dict:
    p, um, mm = r.inputs, 1e6, 1e3
    s_guide = 1.5 * r.delta_mid if s_guide is None else s_guide
    tipdist = float(np.linalg.norm(r.tip))
    half = tipdist + p.r_d + 2.5 * r.delta_mid                   # block shows the slot and the part tip
    data = []

    # machined block: closed triangle meshes of the part and the scrap side
    xs, zs, part, scrap = machined_fields(r, half)
    d1, d2 = model.leg_directions(p.angle_deg)
    into_part = (d2 - d1) / np.linalg.norm(d2 - d1)               # from C towards the part tip
    mesh_style = dict(type="mesh3d", scene="scene", flatshading=True, hoverinfo="name",
                      lighting=dict(ambient=0.5, diffuse=0.8, specular=0.15, roughness=0.6, fresnel=0.0,
                                    facenormalsepsilon=0, vertexnormalsepsilon=0),
                      lightposition=dict(x=1e5, y=2e5, z=3e5))
    for F, anchor, name, colour, extra in ((part, into_part, "part (machined)", "#7d889c", {}),
                                           (scrap, -into_part, "scrap side", "#d8c7a8", dict(visible=False))):
        V, T = solid_mesh(xs, zs, F, anchor)
        trace = dict(mesh_style, x=_r(V[:, 0] * um, 2), y=_r(V[:, 1] * um, 2), z=_r(V[:, 2] * mm, 3),
                     i=T[:, 0].tolist(), j=T[:, 1].tolist(), k=T[:, 2].tolist(),
                     color=colour, opacity=1.0, name=name, **extra)
        if name.startswith("part"):
            dev = _edge_deviation(r, V) * um
            top = max(1.0, float(np.ceil(max(sec["edge"] for sec in r.sections) * um * 1.2)))
            trace.pop("color")
            trace.update(intensity=_r(dev, 2), cmin=0.0, cmax=top,
                         colorscale=[[0, "#7d889c"], [0.15, "#9a8fa0"], [0.5, "#e08a5a"], [1, "#c0392b"]],
                         showscale=True, colorbar=dict(title=dict(text="material removed beyond<br>the ideal edge [µm]", side="top"),
                                                       orientation="h", x=0.32, y=0.13, len=0.36, thickness=12),
                         hovertemplate="removed beyond ideal edge: %{intensity:.2f} µm<extra>part</extra>")
        data.append(trace)
    I_SCRAP = 1

    # programmed path on the top face, corner line through C
    keep = np.all(np.abs(r.guide) <= half, axis=1)
    g = r.guide[keep] * um
    zt, zb = (p.c + p.H) * mm, p.c * mm
    data.append(dict(type="scatter3d", mode="lines", x=_r(g[:, 0]), y=_r(g[:, 1]), z=[zt + 0.15] * len(g),
                     line=dict(color="black", width=3), name="programmed path (top face)", scene="scene"))
    data.append(dict(type="scatter3d", mode="lines+text", x=[0, 0], y=[0, 0], z=[0, p.L * mm],
                     line=dict(color="black", width=2, dash="dot"), text=["", "C"], textposition="top center",
                     name="corner line through C", scene="scene"))

    # wire paths at the five heights
    for sec in r.sections:
        col, ls = model.SECTION_STYLE[sec["name"]]
        path = sec["path"][keep] * um
        data.append(dict(type="scatter3d", mode="lines", x=_r(path[:, 0]), y=_r(path[:, 1]),
                         z=[round(sec["z"] * mm, 3)] * len(path),
                         line=dict(color=col, width=5, dash="dash" if ls == "--" else "solid"),
                         name=f"{sec['name']}: e = {sec['error'] * um:.1f} µm", scene="scene",
                         hovertemplate=f"{sec['name']}<br>lag {sec['lag'] * um:.1f} µm<br>corner error {sec['error'] * um:.1f} µm<extra></extra>"))

    # wire and guides at the chosen guide position (moved by the slider)
    wire, G = _wire_at(r, s_guide)
    zw = np.linspace(0.0, p.L, 121)
    I_WIRE = len(data)
    data.append(dict(type="scatter3d", mode="lines", x=_r(wire[:, 0] * um), y=_r(wire[:, 1] * um), z=_r(zw * mm, 3),
                     line=dict(color=model.WIRE_COLOUR, width=9), name="wire (at guide position)", scene="scene"))
    data.append(dict(type="scatter3d", mode="markers+text", x=_r([G[0] * um] * 2), y=_r([G[1] * um] * 2), z=[0, p.L * mm],
                     marker=dict(symbol="square", size=7, color="#333"), text=["lower guide", "upper guide"],
                     textposition="middle right", name="guides", scene="scene"))
    steps = []
    s_values = np.linspace(-0.9 * half, 0.9 * half, 41)
    s_values = np.sort(np.append(s_values, s_guide))
    active = int(np.argmin(np.abs(s_values - s_guide)))
    for sv in s_values:
        w_, g_ = _wire_at(r, sv)
        steps.append(dict(method="restyle", label=f"{sv * um:+.0f}",
                          args=[dict(x=[_r(w_[:, 0] * um), _r([g_[0] * um] * 2)], y=[_r(w_[:, 1] * um), _r([g_[1] * um] * 2)]),
                                [I_WIRE, I_WIRE + 1]]))

    # lag through the height
    zz = np.linspace(0, p.L, 201)
    data.append(dict(type="scatter", x=_r(model.lag(p, zz) * um), y=_r(zz * mm), mode="lines", xaxis="x", yaxis="y",
                     line=dict(color=model.WIRE_COLOUR, width=3), name="lag y(z)", showlegend=False))
    for sec in r.sections:
        col, _ = model.SECTION_STYLE[sec["name"]]
        data.append(dict(type="scatter", x=[round(sec["lag"] * um, 2)], y=[round(sec["z"] * mm, 2)], mode="markers+text",
                         text=[f"{sec['lag'] * um:.1f} µm"], textposition="middle right", marker=dict(color=col, size=9),
                         xaxis="x", yaxis="y", showlegend=False, hoverinfo="skip"))

    # top view, zoomed on C
    (bx0, by0), (bx1, by1) = r.box[0] * um, r.box[1] * um
    kb = np.all((r.guide >= r.box[0]) & (r.guide <= r.box[1]), axis=1)
    data.append(dict(type="scatter", x=_r(r.guide[kb][:, 0] * um), y=_r(r.guide[kb][:, 1] * um), mode="lines",
                     line=dict(color="black", width=2), xaxis="x2", yaxis="y2", showlegend=False, hoverinfo="skip"))
    for sec in r.sections:
        col, ls = model.SECTION_STYLE[sec["name"]]
        path = sec["path"][kb] * um
        cp = sec["closest"] * um
        data.append(dict(type="scatter", x=_r(path[:, 0]), y=_r(path[:, 1]), mode="lines", xaxis="x2", yaxis="y2",
                         line=dict(color=col, width=2.5, dash="dash" if ls == "--" else "solid"), showlegend=False,
                         hovertemplate=f"{sec['name']}: e = {sec['error'] * um:.1f} µm<extra></extra>"))
        data.append(dict(type="scatter", x=[0, round(cp[0], 2)], y=[0, round(cp[1], 2)], mode="lines", xaxis="x2", yaxis="y2",
                         line=dict(color=col, width=1), showlegend=False, hoverinfo="skip"))
    data.append(dict(type="scatter", x=[0], y=[0], mode="markers+text", text=["C"], textposition="top right",
                     marker=dict(color="black", size=7), xaxis="x2", yaxis="y2", showlegend=False, hoverinfo="skip"))

    axis3 = lambda title, rng: dict(title=dict(text=title), range=rng, backgroundcolor="#f4f4f4", gridcolor="#ddd")
    layout = dict(
        title=dict(text=(f"Corner error model: θ = {p.angle_deg:g}°, H = {p.H_mm:g} mm, T = {p.T_N:g} N, F = {p.F_eff * 1e3:.1f} mN"
                         f" &nbsp;·&nbsp; δ_mid = {r.delta_mid * um:.1f} µm, e_mid = {r.section('Mid-height')['error'] * um:.1f} µm"),
                   x=0.01, font=dict(size=18)),
        font=dict(family=FONT_FAMILY, size=14, color="#222"),
        hoverlabel=dict(font=dict(family=FONT_FAMILY, size=14)),
        scene=dict(domain=dict(x=[0, 0.64], y=[0.13, 0.95]),
                   xaxis=axis3("x [µm]", [-half * um, half * um]), yaxis=axis3("y [µm]", [-half * um, half * um]),
                   zaxis=axis3("height from lower guide [mm]", [0, p.L * mm]),
                   aspectmode="manual", aspectratio=dict(x=1, y=1, z=1.15),
                   camera=dict(eye=dict(x=1.25, y=1.45, z=1.25))),
        xaxis=dict(domain=[0.7, 1.0], anchor="y", title=dict(text="lag behind the guide [µm]"), range=[-1, 1.5 * r.delta_mid * um]),
        yaxis=dict(domain=[0.58, 0.95], anchor="x", title=dict(text="height [mm]"), range=[0, p.L * mm]),
        xaxis2=dict(domain=[0.7, 1.0], anchor="y2", title=dict(text="x [µm]"), range=[bx0, bx1]),
        yaxis2=dict(domain=[0.0, 0.40], anchor="x2", title=dict(text="y [µm]"), range=[by0, by1], scaleanchor="x2"),
        shapes=[dict(type="rect", xref="x", yref="y", x0=-1, x1=1.5 * r.delta_mid * um, y0=p.c * mm, y1=(p.c + p.H) * mm,
                     fillcolor="#e6e6e6", line=dict(width=0), layer="below")],
        annotations=[
            dict(text="Lag through the height (side view)", x=0.85, y=0.995, xref="paper", yref="paper", showarrow=False),
            dict(text="Top view at C (looking down the wire)", x=0.85, y=0.445, xref="paper", yref="paper", showarrow=False),
            dict(text=("Grey solid: machined part (the slot, width 2r_d, is the gap round the wire path).<br>"
                       "x, y in µm, height in mm. Drag to rotate, scroll to zoom, hover for values."),
                 x=0.0, y=-0.075, xref="paper", yref="paper", xanchor="left", showarrow=False, align="left",
                 font=dict(size=12, color="#555")),
        ],
        legend=dict(x=0.0, y=0.9, bgcolor="rgba(255,255,255,0.85)", font=dict(size=13)),
        updatemenus=[dict(type="buttons", direction="right", x=0.0, y=0.965, xanchor="left", yanchor="bottom",
                          pad=dict(r=4, t=0), showactive=True, active=0, font=dict(size=13),
                          buttons=[dict(label="Part only", method="restyle", args=[dict(visible=[False]), [I_SCRAP]]),
                                   dict(label="Part + scrap side", method="restyle", args=[dict(visible=[True], opacity=[1.0]), [I_SCRAP]]),
                                   dict(label="Scrap see-through", method="restyle", args=[dict(visible=[True], opacity=[0.3]), [I_SCRAP]])])],
        sliders=[dict(active=active, steps=steps, x=0.03, len=0.58, y=0.1, yanchor="top", pad=dict(t=0, b=0),
                      currentvalue=dict(prefix="Guide position along the path from C [µm]: ", font=dict(size=14)),
                      font=dict(size=11), ticklen=4)],
        margin=dict(l=10, r=20, t=60, b=95),
        paper_bgcolor="white",
    )
    return dict(data=data, layout=layout)


def _plotly_script() -> str:
    """Inline plotly.js so the file works offline: from the plotly package, else from plotly.min.js
    next to this script, else fall back to the CDN (needs internet to view)."""
    try:
        from plotly.offline import get_plotlyjs
        return f"<script>{get_plotlyjs()}</script>"
    except Exception:
        local = Path(__file__).with_name("plotly.min.js")
        if local.exists():
            return f"<script>{local.read_text(encoding='utf-8')}</script>"
        return f'<script src="{CDN}" charset="utf-8"></script>'


def write_html(r: model.Result, path: str | Path, s_guide: float | None = None) -> Path:
    fig = build_figure(r, s_guide)
    path = Path(path)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wire-EDM corner error modelling v11 1 Nov 2026</title>
{_plotly_script()}
<style>html,body{{margin:0;height:100%;font-family:'Times New Roman',Times,serif}}#fig{{width:100vw;height:100vh}}</style>
</head><body><div id="fig"></div>
<script>
var BASE = {json.dumps(fig)};
// Font sizes are written for a 1000-px-high window; scale them with the window (and so the screen).
function scaleFonts(o, k) {{
  if (Array.isArray(o)) {{ o.forEach(function (v) {{ scaleFonts(v, k); }}); return; }}
  if (o === null || typeof o !== 'object') return;
  Object.keys(o).forEach(function (key) {{
    var v = o[key];
    if ((key === 'font' || key === 'textfont') && v && typeof v.size === 'number') v.size = Math.round(v.size * k * 10) / 10;
    scaleFonts(v, k);
  }});
}}
function draw() {{
  var k = Math.min(Math.max(window.innerHeight / 1000, 0.85), 2.2);
  var fig = JSON.parse(JSON.stringify(BASE));
  scaleFonts(fig, k);
  fig.data.forEach(function (t) {{ if (t.mode && t.mode.indexOf('text') >= 0 && !t.textfont) t.textfont = {{size: Math.round(13 * k)}}; }});
  var gd = document.getElementById('fig');
  if (gd.layout && gd.layout.scene) fig.layout.scene.camera = gd.layout.scene.camera;   // keep the view on resize
  Plotly.react(gd, fig.data, fig.layout, {{responsive: true, displaylogo: false}});
}}
var timer;
window.addEventListener('resize', function () {{ clearTimeout(timer); timer = setTimeout(draw, 200); }});
draw();
</script></body></html>"""
    path.write_text(html, encoding="utf-8")
    return path


def open_in_browser(r: model.Result, path: str | Path | None = None, s_guide: float | None = None) -> Path:
    path = Path(path) if path else Path(__file__).with_name("corner_view_v11.html")
    write_html(r, path, s_guide)
    webbrowser.open(path.resolve().as_uri())
    return path

html_view = sys.modules[__name__]


# =============================================================================== WINDOW
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


FIELDS = [
    ("angle_deg", "Corner angle θ", "deg"),
    ("H_mm", "Workpiece thickness H", "mm"),
    ("c_mm", "Clearance face–guide c", "mm"),
    ("T_N", "Wire tension T", "N"),
    ("F_N", "Spark force F", "N"),
    ("d_mm", "Wire diameter d", "mm"),
    ("gap_um", "Spark gap g", "µm"),
]


FONT = "Times New Roman"


def parse_number(text: str, label: str) -> float:
    """Read a number typed in an entry box. Accepts a comma as the decimal mark and ignores a
    unit typed after the number (for example '30,7' or '30.7 µm'). Raises ValueError naming the box."""
    import re
    t = (text or "").strip().replace(",", ".")
    m = re.match(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", t)
    if not t:
        raise ValueError(f"{label} is empty. Enter a number.")
    if not m:
        raise ValueError(f"{label}: '{text.strip()}' is not a number.")
    return float(m.group(0))


def _dpi_aware():
    """Windows: render at the real screen resolution (otherwise Windows stretches a blurry bitmap)."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


class App(tk.Tk):
    def __init__(self):
        _dpi_aware()
        super().__init__()
        self.title(APP_TITLE)
        self._setup_scaling()
        self.vars = {}
        self.labels = {}
        self.result = None
        self._redraw_job = None

        left = ttk.Frame(self, padding=10)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=(0, 6, 6, 6))
        right.pack(side="left", fill="both", expand=True)

        box = ttk.LabelFrame(left, text="Inputs", padding=8)
        box.pack(fill="x")
        defaults = model.Inputs()
        for i, (key, text, unit) in enumerate(FIELDS):
            ttk.Label(box, text=text).grid(row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=f"{getattr(defaults, key):g}")
            ent = ttk.Entry(box, textvariable=var, width=10, justify="right")
            ent.grid(row=i, column=1, padx=6)
            ttk.Label(box, text=unit).grid(row=i, column=2, sticky="w")
            self.vars[key] = var
            self.labels[key] = text
            if key == "F_N":
                self.F_entry = ent

        box = ttk.LabelFrame(left, text="Spark force", padding=8)
        box.pack(fill="x", pady=(8, 0))
        self.force_var = tk.StringVar(value="fixed")
        ttk.Radiobutton(box, text="Fixed F (entered above; 0.037 N = Chen 2014)", value="fixed",
                        variable=self.force_var, command=self._force_changed).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(box, text="Process law (feed, pulse, voltage, tension)", value="law",
                        variable=self.force_var, command=self._force_changed).grid(row=1, column=0, columnspan=3, sticky="w")
        self.law_entries = []
        for i, (key, text, unit) in enumerate((("feed_mm_min", "Feed v_f", "mm/min"), ("t_on_us", "Pulse-on t_on", "µs"),
                                                ("t_off_us", "Pulse-off t_off", "µs"), ("V_peak", "Peak voltage V_p", "V"),
                                                ("K_N", "Machine constant K", "N"))):
            ttk.Label(box, text=text).grid(row=2 + i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=f"{getattr(defaults, key):g}")
            ent = ttk.Entry(box, textvariable=var, width=10, justify="right")
            ent.grid(row=2 + i, column=1, padx=6)
            ttk.Label(box, text=unit).grid(row=2 + i, column=2, sticky="w")
            self.vars[key] = var
            self.labels[key] = text
            self.law_entries.append(ent)
        ttk.Label(box, text="Measured straight lag").grid(row=7, column=0, sticky="w", pady=2)
        self.lag_var = tk.StringVar(value="")
        ent = ttk.Entry(box, textvariable=self.lag_var, width=10, justify="right")
        ent.grid(row=7, column=1, padx=6)
        ttk.Label(box, text="µm").grid(row=7, column=2, sticky="w")
        btn = ttk.Button(box, text="Set K from this lag", command=self._fit_K)
        btn.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        self.law_entries += [ent, btn]
        ttk.Label(box, text="F grows with the removal rate v_f·H.\nK = 0.0701 N: Chen 2014 machine (L27 fit).", foreground="#555", justify="left").grid(
            row=9, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self._force_changed()

        box = ttk.LabelFrame(left, text="Corner model", padding=8)
        box.pack(fill="x", pady=(8, 0))
        self.model_var = tk.StringVar(value="pursuit")
        ttk.Radiobutton(box, text="Pursuit (wire trails along its own travel)", value="pursuit",
                        variable=self.model_var, command=self._model_changed).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(box, text="Discharge-circle force (sparks on the uncut arc)", value="discharge",
                        variable=self.model_var, command=self._model_changed).grid(row=1, column=0, columnspan=3, sticky="w")
        self.extra_entries = []
        for i, (key, text, unit, val) in enumerate((("conc", "Discharge concentration c", "0–1", "0"),
                                                    ("k_g_um", "Gap growth k_g", "µm", "0"))):
            ttk.Label(box, text=text).grid(row=2 + i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=val)
            ent = ttk.Entry(box, textvariable=var, width=10, justify="right")
            ent.grid(row=2 + i, column=1, padx=6)
            ttk.Label(box, text=unit).grid(row=2 + i, column=2, sticky="w")
            self.vars[key] = var
            self.labels[key] = text
            self.extra_entries.append(ent)
        self._model_changed()
        ttk.Button(left, text="Calculate", command=self.calculate).pack(fill="x", pady=(8, 4))
        ttk.Button(left, text="Open interactive 3-D view in browser", command=self.open_browser).pack(fill="x", pady=(0, 8))
        self.bind("<Return>", lambda e: self.calculate())

        box = ttk.LabelFrame(left, text="Guide position for the 3-D wire", padding=8)
        box.pack(fill="x")
        self.pos = tk.DoubleVar(value=1.5)
        scale = ttk.Scale(box, from_=-3.0, to=3.0, variable=self.pos, orient="horizontal", length=self.px(260),
                          command=lambda v: self._slider_moved())
        scale.pack(fill="x")
        self.pos_text = tk.StringVar()
        ttk.Label(box, textvariable=self.pos_text, foreground="#555").pack(anchor="w")

        tframe = ttk.Frame(left)
        tframe.pack(fill="both", expand=True, pady=(8, 0))
        self.tree = ttk.Treeview(tframe, columns=("q", "v"), show="headings", height=12)
        sb = ttk.Scrollbar(tframe, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.tree.heading("q", text="Quantity")
        self.tree.heading("v", text="Value")
        self.tree.column("q", width=self.px(260))
        self.tree.column("v", width=self.px(150), anchor="e")
        self.tree.pack(side="left", fill="both", expand=True)

        model.set_style(self.plot_pt)
        self.fig = Figure(figsize=(11, 7.6), dpi=self.dpi)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        NavigationToolbar2Tk(self.canvas, right).pack(side="top", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.calculate()

    def _setup_scaling(self):
        """Size fonts and the window from the screen: DPI sets points to pixels, and on large
        screens (more than 1080 logical pixels high) the text grows in proportion."""
        self.dpi = float(self.winfo_fpixels("1i"))                    # pixels per inch
        self.tk.call("tk", "scaling", self.dpi / 72.0)                # font points follow the DPI
        logical_h = self.winfo_screenheight() * 96.0 / self.dpi       # height in 96-dpi pixels
        self.ui_scale = min(max(logical_h / 1080.0, 1.0), 1.8)
        base = round(11 * self.ui_scale)
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkCaptionFont",
                     "TkSmallCaptionFont", "TkIconFont", "TkTooltipFont"):
            try:
                tkfont.nametofont(name).configure(family=FONT, size=base)
            except tk.TclError:
                pass
        tkfont.nametofont("TkHeadingFont").configure(weight="bold")
        line = tkfont.nametofont("TkDefaultFont").metrics("linespace")
        ttk.Style(self).configure("Treeview", rowheight=int(line * 1.35))
        self.plot_pt = 10.0 * self.ui_scale
        w, h = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{int(w * 0.9)}x{int(h * 0.85)}+{int(w * 0.05)}+{int(h * 0.04)}")

    def px(self, n):
        """Scale a pixel size designed for a 96-dpi, 1080-pixel-high screen."""
        return int(n * self.dpi / 96.0 * self.ui_scale)

    def _slider_moved(self):
        self._pos_label()
        if self._redraw_job:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(150, self.redraw)              # redraw once the slider pauses

    def _force_changed(self):
        law = self.force_var.get() == "law"
        for ent in self.law_entries:
            ent.configure(state="normal" if law else "disabled")
        self.F_entry.configure(state="disabled" if law else "normal")

    def _inputs(self):
        vals = {k: parse_number(v.get(), self.labels.get(k, k)) for k, v in self.vars.items()}
        return model.Inputs(**vals, corner_model=self.model_var.get(), force_mode=self.force_var.get())

    def _fit_K(self):
        try:
            if not self.lag_var.get().strip():
                raise ValueError("Enter the measured straight-cut lag (µm) in the box above the button, "
                                 "for the settings currently entered, then press the button.")
            K = model.K_from_lag(self._inputs(), parse_number(self.lag_var.get(), "Measured straight lag"))
        except ValueError as exc:
            messagebox.showerror("Input error", str(exc))
            return
        self.vars["K_N"].set(f"{K:.4g}")
        self.calculate()

    def _model_changed(self):
        state = "normal" if self.model_var.get() == "discharge" else "disabled"
        for ent in self.extra_entries:
            ent.configure(state=state)

    def _pos_label(self):
        if self.result is None:
            return
        s = self.pos.get() * self.result.delta_mid * 1e6
        where = "before" if s < 0 else "past"
        self.pos_text.set(f"guide {abs(s):.0f} µm {where} the corner C ({self.pos.get():+.1f} × δ_mid)")

    def calculate(self):
        try:
            inputs = self._inputs()
            self.config(cursor="watch")
            self.update_idletasks()
            self.result = model.run(inputs)
        except ValueError as exc:
            self.config(cursor="")
            messagebox.showerror("Input error", str(exc))
            return
        self.config(cursor="")
        self.tree.delete(*self.tree.get_children())
        for q, v in model.summary(self.result):
            self.tree.insert("", "end", values=(q, v))
        self._pos_label()
        self.redraw()

    def open_browser(self):
        """Write corner_view.html next to this script and open it (machined block, wire, sections)."""
        if self.result is None:
            return
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            path = html_view.open_in_browser(self.result, s_guide=self.pos.get() * self.result.delta_mid)
        except Exception as exc:                      # file write or browser launch failed
            messagebox.showerror("Browser view", str(exc))
        else:
            self.title(f"{APP_TITLE} — view saved to {path.name}")
        finally:
            self.config(cursor="")

    def redraw(self):
        """Redraw the plots, keeping the 3-D orientation the user set. Zoom and pan are kept too,
        unless a new calculation changed the result (then the limits fit the new result)."""
        if self.result is None:
            return
        same = self.result is getattr(self, "_drawn_result", None)
        view = model.get_view(self.fig, with_limits=same)
        model.plot_all(self.result, fig=self.fig, s_guide=self.pos.get() * self.result.delta_mid, view=view)
        self._drawn_result = self.result
        self.canvas.draw_idle()


if __name__ == "__main__":
    App().mainloop()
