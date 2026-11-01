"""Step 2 of 3: draw the 11 paper figures (PDF + 600 dpi PNG) from paper_data.json into figs/.  python figures.py"""
import json, os, numpy as np, matplotlib; matplotlib.use("Agg")
from pathlib import Path
HERE = Path(__file__).resolve().parent
import matplotlib.pyplot as plt
d = json.load(open(HERE / "paper_data.json")); OUT = str(HERE / "figs"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "in", "ytick.direction": "in",
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600})
SC, DC = 3.5, 7.25
BLK, ORA, SKY, GRN, BLU, VER, PUR, GRY = "#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#8c8c8c"
CP, CD = BLU, VER                       # pursuit, discharge
def lab(ax, s, x=-0.16, y=1.02): ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight"); plt.close(fig)
def parity(ax, hi):
    x = np.array([0, hi]); ax.plot(x, x, "k-", lw=0.6); ax.fill_between(x, 0.8 * x, 1.2 * x, color="0.92", lw=0, zorder=0, label="±20 %")
    ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
SEC_C = {"Top face": VER, "Upper quarter": ORA, "Mid-height": BLU, "Lower quarter": ORA, "Bottom face": VER}

# Fig 1: lag profile and section paths
R = d["R1"]; fig, (a, b) = plt.subplots(1, 2, figsize=(DC, 2.7), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.35))
a.axhspan(R["c"], R["c"] + R["H"], color="0.92", lw=0, label="workpiece")
a.plot(R["y_um"], R["z_mm"], color=GRN, lw=1.4, label="wire lag $y(z)$")
for s in R["sections"]:
    a.plot(s["lag"], s["z"], "o", ms=3.5, color=SEC_C[s["name"]], zorder=3)
a.annotate(f"$\\delta_{{mid}}$ = {R['dmid']:.1f} µm", (R["dmid"], R["c"] + R["H"] / 2), xytext=(-62, 14), textcoords="offset points", fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.5))
a.annotate(f"$\\delta_{{face}}$ = {R['dface']:.1f} µm", (R["dface"], R["c"] + R["H"]), xytext=(-60, 10), textcoords="offset points", fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.5))
a.set_xlabel("lag behind the guide line [µm]"); a.set_ylabel("height above the lower guide $z$ [mm]"); a.set_xlim(0, 36); a.set_ylim(0, R["L"])
a.legend(loc="lower right"); lab(a, "(a)")
G = np.array(R["guide"]); b.plot(G[:, 0], G[:, 1], "k-", lw=0.9, label="programmed path")
done = set()
for s in R["sections"]:
    key = s["name"].split()[-1] if s["name"] != "Mid-height" else "mid"
    name = {"face": "faces", "quarter": "quarters", "mid": "mid-height"}[key]
    if name in done: continue
    done.add(name); P = np.array(s["path"])
    b.plot(P[:, 0], P[:, 1], "-", color=SEC_C[s["name"]], lw=1.2, label=f"{name}: $e$ = {s['e']:.1f} µm")
b.plot(0, 0, "ko", ms=3); b.text(3, 3, "C", fontsize=8, fontweight="bold")
b.set_xlim(-90, 15); b.set_ylim(-90, 12); b.set_aspect("equal"); b.set_xlabel("$x$ [µm]"); b.set_ylabel("$y$ [µm]")
b.legend(loc="lower left"); lab(b, "(b)", x=-0.2)
save(fig, "Fig01_lag_profile_and_section_paths")

# Fig 2: kappa vs angle
K = d["R2"]; fig, ax = plt.subplots(figsize=(SC, 2.6))
ax.plot(K["theta"], K["kappa_p"], "-o", color=CP, ms=3, label="pursuit")
ax.plot(K["theta"], K["kappa_d"], "-s", color=CD, ms=3, label="discharge-circle force")
ch = K["chen"]; ax.plot(ch["theta"], np.array(ch["e"]) / np.array(ch["lag"]), "k*", ms=7, label="Chen et al. [3], $e/\\delta$ measured")
ax.set_xlabel("corner angle $\\theta$ [°]"); ax.set_ylabel("$\\kappa = e/\\delta_{mid}$"); ax.set_xlim(25, 170); ax.set_ylim(0, 1.08)
ax.legend(loc="lower left"); save(fig, "Fig02_kappa_vs_angle")

# Fig 3: discharge force through the corner
fig, (a, b) = plt.subplots(1, 2, figsize=(DC, 2.5), gridspec_kw=dict(wspace=0.3)); cols = {"60": PUR, "90": VER, "120": BLU}
for th, c in cols.items():
    r = d["R3"][th]; a.plot(r["s"], r["f"], "-", color=c, lw=1.2, label=f"$|f|$, {th}°"); a.plot(r["s"], r["S"], ":", color=c, lw=1.2, label=f"$S/S_0$, {th}°")
    b.plot(r["s"], r["rot"], "-", color=c, lw=1.2, label=f"{th}°")
for ax in (a, b): ax.set_xlim(-2, 12); ax.axvline(0, color="0.7", lw=0.5); ax.set_xlabel("guide position past C, $s/\\delta_{mid}$")
a.set_ylabel("relative to a straight cut"); a.set_ylim(0.65, 1.02); a.legend(ncol=3, loc="lower right", columnspacing=0.8)
b.set_ylabel("force rotation $\\alpha$ from $-v$ [°]"); b.legend(loc="upper right"); lab(a, "(a)"); lab(b, "(b)")
save(fig, "Fig03_discharge_force_through_corner")

# Fig 4: force-law fits
F4 = d["R4"]; bb, se = np.array(F4["b"]), np.array(F4["se"]); res = np.array(F4["res"])
names = [("removal rate $v_f H$ [mm²/min]", "a"), ("pulse-on $t_{on}$ [µs]", "b"), ("pulse-off $t_{off}$ [µs]", "c"), ("peak voltage $V_p$ [V]", "d"), ("tension $T$ [N]", "g")]
fig, axs = plt.subplots(1, 5, figsize=(DC, 2.0), sharey=True, gridspec_kw=dict(wspace=0.12))
for j, (ax, (nm, sym)) in enumerate(zip(axs, names), start=1):
    v = np.array(F4["vals"][j - 1]); ref = F4["refs"][j - 1]; xj = np.log(v / ref); pr = np.exp(res + bb[j] * xj)
    jit = 1 + np.random.default_rng(j).uniform(-0.015, 0.015, 27)
    ax.scatter(v * jit, pr, s=5, color=SKY, edgecolor="none", zorder=3)
    lv = np.array(sorted(set(v))); ax.plot(lv, [np.exp(np.log(pr[v == u]).mean()) for u in lv], "k*", ms=5, zorder=4)
    xx = np.linspace(lv.min() * 0.93, lv.max() * 1.05, 100); ax.plot(xx, (xx / ref) ** bb[j], "-", color=VER, lw=1.2)
    for k in (-1, 1): ax.plot(xx, (xx / ref) ** (bb[j] + k * se[j]), ":", color=VER, lw=0.7)
    ax.axhline(1, color="0.85", lw=0.5, zorder=0); ax.set_xticks(lv); ax.set_xlabel(nm, fontsize=7)
    ax.text(0.04, 0.96, f"${sym}$ = {bb[j]:.2f} ± {se[j]:.2f}", transform=ax.transAxes, fontsize=7, va="top")
axs[0].set_ylabel("force multiplier"); axs[0].set_ylim(0.4, 1.95)
save(fig, "Fig04_force_law_fits")

# Fig 5: lag shape, Liang
fig, ax = plt.subplots(figsize=(SC, 2.8)); cl = {"30": BLU, "50": GRN, "70": VER}
for key, pts in d["V1"].items():
    H = float(key.split("-")[1]); P = np.array(pts); c = cl[key.split("-")[1]]; csm = key.endswith("CSM")
    ax.plot(P[:, 1] / P[:, 1].max(), P[:, 0] / H, "s" if csm else "o", ms=2.6, color=c, mfc="none" if csm else c, mew=0.6, alpha=0.85,
            label=f"$H$ = {int(H)} mm, {key[-3:]}")
u = np.linspace(0, 1, 101); ax.plot(4 * u * (1 - u), u, "k-", lw=1.4, label="model (string, uniform load)")
ax.set_xlabel("deflection / maximum deflection"); ax.set_ylabel("height in the workpiece $z/H$"); ax.set_xlim(0, 1.05); ax.set_ylim(0, 1)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=6.5, handletextpad=0.3, columnspacing=0.8); save(fig, "Fig05_lag_shape_Liang")

# Fig 6: Chen corner error vs angle (F from Chen's lag)
fig, ax = plt.subplots(figsize=(SC, 2.6)); th = np.array(K["theta"])
ax.plot(th, 30.7 * np.array(K["kappa_p"]), "-", color=CP, label="pursuit, mid-height")
ax.plot(th, 30.7 * np.array(K["kappa_d"]), "-", color=CD, label="discharge-circle force, mid-height")
for r in d["V2"]:
    ax.vlines(r["theta"] - 2, r["pursuit"]["face"], r["pursuit"]["mid"], color=CP, lw=3, alpha=0.35)
    ax.vlines(r["theta"] + 2, r["discharge"]["face"], r["discharge"]["mid"], color=CD, lw=3, alpha=0.35)
ax.plot([], [], color="0.6", lw=3, alpha=0.5, label="face to mid-height range")
ax.plot([r["theta"] for r in d["V2"]], [r["meas"] for r in d["V2"]], "k*", ms=7, label="Chen et al. [3], measured")
ax.set_xlabel("corner angle $\\theta$ [°]"); ax.set_ylabel("corner path error $e$ [µm]"); ax.set_xlim(25, 170); ax.set_ylim(0, 34)
ax.legend(loc="lower left"); save(fig, "Fig06_Chen_corner_error_vs_angle")

# Fig 7: force-law validation
fig, (a, b) = plt.subplots(1, 2, figsize=(DC, 3.3), gridspec_kw=dict(wspace=0.3)); mk = {"45": ("^", GRY), "90": ("o", BLU), "135": ("s", VER)}
parity(a, 125)
for t, (m_, c) in mk.items():
    M, P = np.array(d["V3a"][t]["meas"]), np.array(d["V3a"][t]["pred"])
    a.scatter(M, P, marker=m_, s=12, color=c, edgecolor="k", lw=0.3, zorder=3,
              label=f"{t}°{', leave-one-out' if t == '90' else ', predicted'}: {100*np.mean(np.abs(P/M-1)):.0f} %")
a.set_xlabel("measured corner path error [µm]"); a.set_ylabel("predicted corner path error [µm]"); a.legend(loc="upper left", title="Chen et al. [3], 27 L27 runs", title_fontsize=7, fontsize=6.5); lab(a, "(a)")
parity(b, 95)
for t, (m_, c) in mk.items():
    M = np.array([r["meas"][t] for r in d["V3b"]]); P = np.array([r["pred"][t] for r in d["V3b"]])
    b.scatter(M, P, marker=m_, s=16, color=c, edgecolor="k", lw=0.3, zorder=3, label=f"{t}°: {100*np.mean(np.abs(P/M-1)):.0f} %")
b.set_xlabel("measured corner path error [µm]"); b.set_ylabel("predicted corner path error [µm]"); b.legend(loc="upper left", title="Chen et al. [3], 8 held-out runs", title_fontsize=7, fontsize=6.5); lab(b, "(b)")
save(fig, "Fig07_force_law_validation_Chen")

# Fig 8: thickness
fig, (a, b) = plt.subplots(1, 2, figsize=(DC, 2.7), gridspec_kw=dict(wspace=0.3)); O = d["V4"]["obara"]; Li = d["V4"]["liang"]
def thick(ax, rows, meas_keys, ylab, kcal):
    H = [r["H"] for r in rows]
    for k, m_, lb in meas_keys: ax.plot(H, [r[k] for r in rows], "k" + m_, ms=6, ls="none", label=lb, zorder=4)
    ax.plot(H, [r["fixed"] for r in rows], "--", color=GRY, marker="s", ms=3, label="fixed $F$")
    ax.plot(H, [r["feedonly"] for r in rows], "-", color=PUR, marker="v", ms=3, label="$F \\propto v_f^{\\,a}$ (feed only)")
    ax.plot(H, [r["law"] for r in rows], "-", color=GRN, marker="D", ms=3, label="$F \\propto (v_f H)^{a}$ (v9 law)")
    ax.set_xticks(H); ax.set_xticklabels([f"{r['H']}\n({r['v']:g} mm/min)" for r in rows])
    ax.set_xlabel("workpiece thickness $H$ [mm] (feed $v_f$)"); ax.set_ylabel(ylab); ax.set_ylim(0, None)
    ax.text(0.98, 0.97, f"$K$ set at $H$ = {kcal} mm", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color="0.35")
thick(a, O, [("meas", "*", "Obara et al. [5]")], "straight-cut lag $\\delta_{mid}$ [µm]", 40); a.set_ylim(0, 390); a.legend(loc="upper left"); lab(a, "(a)")
thick(b, Li, [("cpa", "o", "Liang and Liao [8], CPA"), ("csm", "s", "Liang and Liao [8], CSM")], "maximum wire bow [µm]", 30); b.set_ylim(0, 215); b.legend(loc="upper left"); lab(b, "(b)")
save(fig, "Fig08_thickness_validation")

# Fig 9: Hsue spark dip
fig, ax = plt.subplots(figsize=(SC, 2.6)); V = d["V5"]; th = [r["theta"] for r in V]
ax.axvspan(25, 80, color="0.94", lw=0, zorder=0); ax.text(28, 96, "acute (outside the study scope)", fontsize=6.5, color="0.4", va="top")
ax.plot(th, [r["meas"] for r in V], "k*-", ms=6, lw=0.8, label="Hsue et al. [1], measured")
ax.plot(th, [r["hsue_model"] for r in V], ":", color="k", marker="x", ms=4, lw=0.8, label="Hsue et al. [1], removal-rate model")
ax.plot(th, [r["rigid_rod"] for r in V], "--", color=GRY, lw=0.8, label="rigid rod, $(1-\\cos\\theta)/2$")
ax.plot(th, [r["rigid"] for r in V], "s-", color=PUR, ms=3, label="this model, rigid wire")
ax.plot(th, [r["lag"] for r in V], "o-", color=VER, ms=3, label="this model, 100 µm lag")
ax.set_xlabel("corner angle $\\theta$ [°]"); ax.set_ylabel("minimum spark rate $S/S_0$ [%]"); ax.set_xlim(25, 155); ax.set_ylim(0, 105)
ax.legend(loc="lower right", fontsize=6.5); save(fig, "Fig09_spark_rate_dip_Hsue")

# Fig 10: tension
T6 = d["V6"]; fig, (a, b) = plt.subplots(1, 2, figsize=(DC, 2.6), gridspec_kw=dict(wspace=0.3))
for key, lk, ls, nm in (("chen_meas90", "law90", "-", "90°"), ("chen_meas135", "law135", "--", "135°")):
    a.plot(T6["T"], T6[key], "k" + ls, marker="*", ms=6, label=f"Chen et al. [3], {nm}")
    a.plot(T6["T"], T6[lk], ls, color=BLU, marker="o", ms=3, label=f"v9 force law, {nm}")
    fx = np.array(T6[key][1]) * 10 / np.array(T6["T"]); a.plot(T6["T"], fx, ls, color=GRY, lw=0.9, label=f"fixed $F$ ($e \\propto 1/T$), {nm}")
a.set_xlabel("wire tension $T$ [N]"); a.set_ylabel("mean corner path error [µm]"); a.set_ylim(0, 112); a.set_xticks([5, 10, 15]); a.legend(ncol=2, loc="upper right", fontsize=6.3); lab(a, "(a)")
for t, m_ in (("90", "*"), ("60", "^"), ("30", "v")):
    w = np.array(T6["wang"][t]); b.plot(T6["wang_T"], w / w[1], "k:", marker=m_, ms=5, lw=0.8, label=f"Wang et al. [4], {t}°")
b.plot(T6["wang_T"], T6["wang_fixed"], "--", color=GRY, marker="s", ms=3, label="fixed $F$ ($e \\propto 1/T$)")
b.plot(T6["wang_T"], T6["wang_law"], "-", color=BLU, marker="o", ms=3, label="v9 force law ($e \\propto T^{-0.16}$)")
b.set_xlabel("wire tension $T$ [N]"); b.set_ylabel("corner error / value at 13.7 N"); b.legend(loc="lower left", fontsize=6.5); lab(b, "(b)")
save(fig, "Fig10_tension_trend")

# Fig 11: Wang edge error
W = d["V7"]; fig, ax = plt.subplots(figsize=(SC, 2.5)); x = np.arange(len(W)); w = 0.2
ax.bar(x - 1.5 * w, [r["meas"] for r in W], w, color="0.25", label="Wang et al. [4], measured")
ax.bar(x - 0.5 * w, [r["pursuit"]["edge"] for r in W], w, color=CP, label="edge error, pursuit")
ax.bar(x + 0.5 * w, [r["discharge"]["edge"] for r in W], w, color=CD, label="edge error, discharge")
ax.bar(x + 1.5 * w, [r["discharge"]["e"] for r in W], w, color="white", edgecolor=CD, hatch="////", lw=0.6, label="path error, discharge")
ax.set_xticks(x); ax.set_xticklabels([r["label"] for r in W], fontsize=7); ax.set_ylabel("error [µm]"); ax.set_ylim(0, 42); ax.legend(loc="upper left", ncol=2, fontsize=6.3)
save(fig, "Fig11_edge_error_Wang")
print("ok")
