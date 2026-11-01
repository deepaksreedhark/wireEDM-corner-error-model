# -*- coding: utf-8 -*-
"""
fit_force_law.py — derives the process force law used by simplified_corner_gui_v11.py
====================================================================================

    F = K (v_f*H/100)^a (t_on/12)^b (t_off/12)^c (V_p/40)^d (T/10)^g

    v_f feed [mm/min], H thickness [mm] (v_f*H = removal rate per unit kerf width, mm^2/min),
    t_on / t_off pulse-on / pulse-off [us], V_p peak voltage [V], T wire tension [N].

Method (methodology rev6, Section 7):
 1. Data: Chen et al. (2014), Int J Adv Manuf Technol 74:917-929, Table 4 — Taguchi L27, 27 runs, 25 mm SKD11,
    0.25 mm brass wire, corner path error at 45, 90 and 135 deg. Tension given in "kgf" is read as N.
 2. For each run, invert the model at 90 deg to get the force it needs:
        e = kappa(90) * delta_mid,   delta_mid = F (2L - H) / 8T   ->   F = 8 T e / (kappa (2L - H))
    with H = 25 mm, L = H + 2c = 45 mm, kappa(90) = 0.663 (pursuit model).
 3. Take logs: ln F = ln K + a ln(v_f H/100) + b ln(t_on/12) + c ln(t_off/12) + d ln(V_p/40) + g ln(T/10),
    and solve by ordinary least squares (27 equations, 6 unknowns).
 4. Check: standard errors and t-values, R^2 and adjusted R^2, leave-one-out error, alternative term sets,
    and the 8 confirmation runs of Chen et al. (Table 8), which are not used in the fit.

Chen's runs are all 25 mm thick, so v_f*H/100 = v_f/4: the data cannot separate feed from thickness. The
removal-rate form is chosen for its physics (sparks per second follow the metal removed per second) and is
supported by the thickness data of Obara et al. (2003) and Liang & Liao (2014) (methodology Section 7.6).

Run:   python fit_force_law.py           Needs: numpy.  Writes force_law_fit_results.json next to this file.
"""
import json, math
from pathlib import Path
import numpy as np

# ------------------------------------------------------------------------------------------------ data
# Chen et al. 2014, Table 4. Columns: run, A water pressure [kg/cm2], B t_on [us], C t_off [us], D V_p [V],
# E feed [mm/min], F tension [kgf, read as N], G wire speed [m/min], e45, e90, e135 [um]
L27 = np.array([
    (1, 3, 8, 8, 30, 3, 5, 5, 22.6, 14.5, 8.2),
    (2, 3, 8, 8, 30, 4, 10, 10, 34.9, 20.6, 11.7),
    (3, 3, 8, 8, 30, 5, 15, 15, 50.8, 28.3, 18.7),
    (4, 3, 12, 12, 40, 3, 5, 5, 40.1, 28.1, 13.3),
    (5, 3, 12, 12, 40, 4, 10, 10, 58.9, 38, 22.8),
    (6, 3, 12, 12, 40, 5, 15, 15, 79.9, 48.9, 36.1),
    (7, 3, 16, 16, 50, 3, 5, 5, 77.6, 53.9, 37.2),
    (8, 3, 16, 16, 50, 4, 10, 10, 86.9, 59.7, 30.7),
    (9, 3, 16, 16, 50, 5, 15, 15, 114.7, 74.5, 53.6),
    (10, 5, 8, 12, 50, 3, 10, 15, 35.1, 19.4, 8.7),
    (11, 5, 8, 12, 50, 4, 15, 5, 50.7, 28.3, 18.9),
    (12, 5, 8, 12, 50, 5, 5, 10, 76.6, 46.6, 32.4),
    (13, 5, 12, 16, 30, 3, 10, 15, 23.3, 13.4, 8.7),
    (14, 5, 12, 16, 30, 4, 15, 5, 44.5, 28, 15.4),
    (15, 5, 12, 16, 30, 5, 5, 10, 62.9, 43, 26),
    (16, 5, 16, 8, 40, 3, 10, 15, 56.3, 41.3, 24.3),
    (17, 5, 16, 8, 40, 4, 15, 5, 71.5, 51, 31),
    (18, 5, 16, 8, 40, 5, 5, 10, 114.1, 78.6, 62.6),
    (19, 7, 8, 16, 40, 3, 15, 10, 18.6, 10.1, 10.7),
    (20, 7, 8, 16, 40, 4, 5, 15, 39.7, 22.5, 9.1),
    (21, 7, 8, 16, 40, 5, 10, 5, 55.9, 32.8, 14.2),
    (22, 7, 12, 8, 50, 3, 15, 10, 43.3, 28.8, 12.7),
    (23, 7, 12, 8, 50, 4, 5, 15, 78.5, 49.7, 37.8),
    (24, 7, 12, 8, 50, 5, 10, 5, 88.9, 57.7, 39.4),
    (25, 7, 16, 12, 30, 3, 15, 10, 49.6, 35.5, 24.5),
    (26, 7, 16, 12, 30, 4, 5, 15, 72.2, 50.7, 35.7),
    (27, 7, 16, 12, 30, 5, 10, 5, 92.2, 63.8, 42.7),
])
# Chen et al. 2014, Table 8 (confirmation runs, held out). Columns: feed, t_on, t_off, V_p, T, e45, e90, e135
CONFIRM = np.array([
    (4, 10, 10, 45, 15, 52.4, 31.2, 20.9),
    (5, 10, 14, 35, 15, 54.3, 32.7, 25.6),
    (5, 14, 10, 35, 10, 79.4, 54.1, 31.5),
    (4, 8, 8, 45, 10, 46.2, 24.3, 18.5),
    (3, 16, 16, 45, 10, 62.3, 46.3, 22.9),
    (5, 8, 11, 30, 15, 48.4, 27.8, 22.1),
    (5, 8, 11, 30, 15, 49.3, 28.3, 22.7),
    (5, 8, 10, 32, 15, 46.7, 29.3, 21.8),
])
H_MM, C_MM = 25.0, 10.0                          # thickness and face-to-guide clearance [mm]
KAPPA = {45: 1.000, 90: 0.663, 135: 0.298}       # pursuit-model kappa(theta) = e / delta_mid
REF = dict(Q=100.0, t_on=12.0, t_off=12.0, V_p=40.0, T=10.0)   # reference settings (Chen's middle levels)


def geo(T_N, H=H_MM, c=C_MM):
    """delta_mid per unit force [m/N] for the string model: (2L - H) / 8T."""
    L = H + 2 * c
    return (2 * L - H) * 1e-3 / (8 * T_N)


def columns(feed, t_on, t_off, V, T, water=None, ws=None, H=H_MM):
    """Log design columns. Removal rate Q = v_f * H [mm^2/min]."""
    cols = {"removal rate v_f*H": np.log(feed * H / REF["Q"]), "pulse-on t_on": np.log(t_on / REF["t_on"]),
            "pulse-off t_off": np.log(t_off / REF["t_off"]), "peak voltage V_p": np.log(V / REF["V_p"]),
            "tension T": np.log(T / REF["T"])}
    if water is not None:
        cols["water pressure"] = np.log(water / 5.0)
        cols["wire speed"] = np.log(ws / 10.0)
    return cols


def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b; n, p = X.shape; s2 = (r @ r) / (n - p)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean())); adj = 1 - (1 - r2) * (n - 1) / (n - p)
    return b, se, r2, adj, r


def loo_error(X, y, T, e90):
    """Leave-one-out: refit on 26 runs, predict the 90 deg error of the 27th. Mean absolute % error."""
    err = []
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        b, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        F = math.exp(X[i] @ b); err.append(KAPPA[90] * F * geo(T[i]) * 1e6 / e90[i] - 1)
    return 100 * np.mean(np.abs(err))


def main():
    water, t_on, t_off, V, feed, T, ws = (L27[:, k] for k in range(1, 8))
    e45, e90, e135 = L27[:, 8], L27[:, 9], L27[:, 10]
    F_needed = e90 * 1e-6 / (KAPPA[90] * geo(T))            # [N] force that reproduces each measured 90 deg error
    y = np.log(F_needed)

    variants = {
        "adopted: removal rate, t_on, t_off, V_p, T": columns(feed, t_on, t_off, V, T),
        "without tension term": {k: v for k, v in columns(feed, t_on, t_off, V, T).items() if k != "tension T"},
        "all seven settings (+ water, wire speed)": columns(feed, t_on, t_off, V, T, water, ws),
    }
    results = {}
    print("Force-law fit to Chen et al. 2014 L27 runs (corner error at 90 deg, pursuit model)\n")
    for name, cols in variants.items():
        X = np.column_stack([np.ones(len(y))] + list(cols.values()))
        b, se, r2, adj, res = ols(X, y)
        loo = loo_error(X, y, T, e90)
        results[name] = dict(K_N=math.exp(b[0]), K_se_N=math.exp(b[0]) * se[0], R2=r2, adjR2=adj, loo_pct=loo,
                             exponents={k: dict(value=b[i + 1], se=se[i + 1], t=b[i + 1] / se[i + 1]) for i, k in enumerate(cols)})
        print(f"--- {name}")
        print(f"    K = {math.exp(b[0])*1e3:.1f} mN   R2 = {r2:.3f}   adjusted R2 = {adj:.3f}   leave-one-out error = {loo:.1f} %")
        for i, k in enumerate(cols):
            print(f"    {k:22s} exponent {b[i+1]:+.3f}  +/- {se[i+1]:.3f}   t = {b[i+1]/se[i+1]:+.1f}")
        print()

    # adopted law: held-out test on the confirmation runs, and the L27 runs at all three angles
    ad = results["adopted: removal rate, t_on, t_off, V_p, T"]
    ex = [ad["exponents"][k]["value"] for k in ("removal rate v_f*H", "pulse-on t_on", "pulse-off t_off", "peak voltage V_p", "tension T")]
    def law(fd, on, off, Vp, TT, H=H_MM):
        return ad["K_N"] * (fd * H / REF["Q"]) ** ex[0] * (on / REF["t_on"]) ** ex[1] * (off / REF["t_off"]) ** ex[2] \
               * (Vp / REF["V_p"]) ** ex[3] * (TT / REF["T"]) ** ex[4]
    print("--- adopted law on Chen's 8 confirmation runs (Table 8, not used in the fit)")
    print("    run  feed t_on t_off  V_p    T |  e45 meas/model |  e90 meas/model | e135 meas/model")
    held = {th: [] for th in (45, 90, 135)}
    for i, (fd, on, off, Vp, TT, m45, m90, m135) in enumerate(CONFIRM, start=1):
        dm = law(fd, on, off, Vp, TT) * geo(TT) * 1e6
        p = {th: KAPPA[th] * dm for th in (45, 90, 135)}
        for th, m in ((45, m45), (90, m90), (135, m135)): held[th].append(p[th] / m - 1)
        print(f"    {i:3d} {fd:5g} {on:4g} {off:5g} {Vp:4g} {TT:4g} | {m45:6.1f} / {p[45]:5.1f} | {m90:6.1f} / {p[90]:5.1f} | {m135:6.1f} / {p[135]:5.1f}")
    for th in (45, 90, 135):
        print(f"    mean absolute error at {th:3d} deg: {100*np.mean(np.abs(held[th])):.0f} %")
    ad["confirmation_mape_pct"] = {th: 100 * float(np.mean(np.abs(held[th]))) for th in held}

    print("\n--- block to paste into the tool (FORCE_LAW_K / FORCE_LAW)")
    print(f"FORCE_LAW_K = {ad['K_N']:.4f}")
    for (k, ref), e_ in zip((("removal_rate", 100.0), ("t_on_us", 12.0), ("t_off_us", 12.0), ("V_peak", 40.0), ("T_N", 10.0)), ex):
        print(f'    ("{k}", {ref}, {e_:.3f}),')

    out = Path(__file__).with_name("force_law_fit_results.json")
    out.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    print(f"\nResults written to {out.name}")


if __name__ == "__main__":
    main()
