"""Step 1 of 3: all model values for the paper figures, computed with simplified_corner_gui_v11.py.
Run with a Python that has tkinter:  python compute.py   (writes paper_data.json, about 10 min)"""
import importlib.util, sys, json, math, time, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("g", str(HERE.parent / "simplified_corner_gui_v11.py"))
g = importlib.util.module_from_spec(spec); sys.modules["g"] = g; spec.loader.exec_module(g)
OLD = json.load(open(HERE / "data" / "val_results.json"))          # measured literature data (already collated)
L = np.array(OLD["L27"], float)
out = {"version": g.VERSION}; t0 = time.time()
um = lambda x: float(x) * 1e6
# R1: default case, pursuit
r = g.run(g.Inputs()); p = r.inputs
z = np.linspace(0, p.L, 301)
out["R1"] = dict(z_mm=(z * 1e3).tolist(), y_um=(g.lag(p, z) * 1e6).tolist(), c=p.c_mm, H=p.H_mm, L=p.L * 1e3,
                 dmid=um(r.delta_mid), dface=um(r.delta_face),
                 sections=[dict(name=s["name"], z=s["z"] * 1e3, lag=um(s["lag"]), e=um(s["error"]),
                                path=(s["path"][::20] * 1e6).tolist()) for s in r.sections],
                 guide=(r.guide[::20] * 1e6).tolist())
print("R1", round(time.time() - t0))
# R2: kappa vs angle, both models (delta 30.7, r_d 150)
angs = [30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
kp, kd = [], []
for th in angs:
    rp = g.run(g.Inputs(angle_deg=th)); kp.append(rp.section("Mid-height")["error"] / rp.delta_mid)
    dc = g.discharge_corner(th, rp.delta_mid, rp.inputs.r_d); kd.append(dc["error"] / rp.delta_mid)
out["R2"] = dict(theta=angs, kappa_p=kp, kappa_d=kd, chen=dict(theta=[45, 90, 135], e=[26.5, 18.4, 8.6], lag=[30.4, 30.7, 30.9]))
print("R2", round(time.time() - t0))
# R3: discharge force through the corner at 60, 90, 120
R3 = {}
for th in (60, 90, 120):
    dc = g.discharge_corner(th, 30.7e-6, 150e-6); f = dc["f"]; fm = np.hypot(*f.T)
    rot = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", -f / np.maximum(fm, 1e-12)[:, None], dc["v"]), -1, 1)))
    k = slice(None, None, 3)
    R3[th] = dict(s=(dc["s"][k] / 30.7e-6).tolist(), f=fm[k].tolist(), S=dc["S_rate"][k].tolist(), rot=rot[k].tolist())
out["R3"] = R3; print("R3", round(time.time() - t0))
# R4 / V3: force-law regression on L27 at 90 deg (removal rate v_f*H, H = 25)
geo = lambda T, H=25e-3, c=10e-3: (2 * (H + 2 * c) - H) / (8 * T)
yv = np.log(L[:, 9] * 1e-6 / 0.663 / geo(L[:, 6]))
vals = [L[:, 5] * 25, L[:, 2], L[:, 3], L[:, 4], L[:, 6]]; refs = [100, 12, 12, 40, 10]
X = np.column_stack([np.ones(27)] + [np.log(v / r_) for v, r_ in zip(vals, refs)])
b, *_ = np.linalg.lstsq(X, yv, rcond=None); res = yv - X @ b
s2 = (res ** 2).sum() / 21; se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X))); r2 = 1 - (res ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
# check against the coefficients in the code
code_exp = [e for _, _, e, _ in g.FORCE_LAW]
out["R4"] = dict(b=b.tolist(), se=se.tolist(), r2=r2, code_K=g.FORCE_LAW_K, code_exp=code_exp,
                 vals=[v.tolist() for v in vals], refs=refs, res=res.tolist())
loo = []
for i in range(27):
    m = np.arange(27) != i; bb, *_ = np.linalg.lstsq(X[m], yv[m], rcond=None); loo.append(math.exp(X[i] @ bb))
loo = np.array(loo); KAP = {45: 1.0, 90: kp[angs.index(90)], 135: kp[angs.index(135)]}
out["V3a"] = {str(th): dict(meas=L[:, col].tolist(), pred=(KAP[th] * loo * geo(L[:, 6]) * 1e6).tolist()) for th, col in ((45, 8), (90, 9), (135, 10))}
# V3b: confirmation runs through the v9 code (process law, pursuit)
conf = json.load(open(HERE / "data" / "force_fit.json"))["conf"]; V3b = []
for cr in conf:
    row = dict(settings=cr, meas={k: v for k, v in cr["e"].items()}, pred={})
    for th in (45, 90, 135):
        rr = g.run(g.Inputs(angle_deg=th, H_mm=25, c_mm=10, T_N=cr["T"], force_mode="law", feed_mm_min=cr["feed"],
                            t_on_us=cr["on"], t_off_us=cr["off"], V_peak=cr["V"]))
        row["pred"][str(th)] = um(rr.section("Mid-height")["error"])
    V3b.append(row)
out["V3b"] = V3b
bl = g.run(g.Inputs(angle_deg=90, H_mm=25, c_mm=10, T_N=10, force_mode="law"))
out["baseline"] = dict(e90=um(bl.section("Mid-height")["error"]), lag=um(bl.delta_mid), meas_e90=18.4, meas_lag=30.7, run5=L[4, 9])
print("V3", round(time.time() - t0))
# V1: Liang shape (data) - model is the parabola; nothing to compute
out["V1"] = OLD["V3"]
# V2: Chen baseline corner error vs angle, F from Chen's own lag at each angle (T 10 N)
V2 = []
for th, lag, meas in ((45, 30.4, 26.5), (90, 30.7, 18.4), (135, 30.9, 8.6)):
    F = 8 * 10 * lag * 1e-6 / (2 * 0.045 - 0.025); row = dict(theta=th, meas=meas, lag=lag)
    for mode in ("pursuit", "discharge"):
        rr = g.run(g.Inputs(angle_deg=th, H_mm=25, c_mm=10, T_N=10, F_N=F, corner_model=mode))
        row[mode] = dict(mid=um(rr.section("Mid-height")["error"]), face=um(rr.section("Top face")["error"]),
                         edge=um(rr.section("Mid-height")["edge"]))
    V2.append(row)
out["V2"] = V2; print("V2", round(time.time() - t0))
# V4: thickness, Obara (lag) and Liang (bow); K from one thickness via the code
ob = []; Kb = g.K_from_lag(g.Inputs(force_mode="law", T_N=9.8, c_mm=10, H_mm=40, feed_mm_min=3.9), 242)
for H, v, m in ((20, 5.6, 113), (40, 3.9, 242), (80, 1.3, 242)):
    q = g.Inputs(force_mode="law", T_N=9.8, c_mm=10, H_mm=H, feed_mm_min=v, K_N=Kb); law = um(g.run(q).delta_mid)
    fixed = 242 * (2 * (H + 20) - H) / (2 * 60 - 40)
    feedonly = fixed * (v / 3.9) ** code_exp[0]
    ob.append(dict(H=H, v=v, meas=m, law=law, fixed=fixed, feedonly=feedonly))
li = []
for H, v, mc, ms in ((30, 4.0, 59.9, 66.0), (50, 2.4, 112.0, 132.0), (70, 1.71, 134.9, 156.0)):
    q = g.Inputs(force_mode="law", T_N=11.8, H_mm=H, feed_mm_min=v)
    bow = q.F_eff * H * 1e-3 / (8 * 11.8); li.append(dict(H=H, v=v, cpa=mc, csm=ms, bow_unit=bow))
s0 = 59.9 / li[0]["bow_unit"]
for d in li:
    d["law"] = d["bow_unit"] * s0; d["fixed"] = 59.9 * d["H"] / 30; d["feedonly"] = d["fixed"] * (d["v"] / 4.0) ** code_exp[0] * (30 / d["H"]) ** 0 * 1.0
    d["feedonly"] = 59.9 * (d["H"] / 30) * (d["v"] / 4.0) ** code_exp[0]
out["V4"] = dict(obara=ob, liang=li); print("V4", round(time.time() - t0))
# V5: Hsue spark dip, r_d 200 µm, lag 100 µm
V5 = []
for row in OLD["hsue"]["rows"]:
    th = row["theta"]; dl = g.discharge_corner(th, 100e-6, 200e-6); dr = g.discharge_corner(th, 100e-6, 200e-6, rigid=True)
    V5.append(dict(theta=th, meas=row["meas"], hsue_model=row["hsue_model"], rigid_rod=row["rigid_rod"],
                   lag=100 * float(dl["S_rate"].min()), rigid=100 * float(dr["S_area"].min())))
    print("  hsue", th, round(V5[-1]["lag"], 1), round(V5[-1]["rigid"], 1), round(time.time() - t0))
out["V5"] = V5
# V6: tension. Chen L27 main effects vs law (averaged over runs) and fixed F; Wang ratios
pred90 = []; pred135 = []
for rrow in L:
    q = g.Inputs(H_mm=25, c_mm=10, T_N=rrow[6], force_mode="law", feed_mm_min=rrow[5], t_on_us=rrow[2], t_off_us=rrow[3], V_peak=rrow[4])
    dm = (2 * q.L - q.H) / (8 * q.T_N) * q.F_eff * 1e6
    pred90.append(KAP[90] * dm); pred135.append(KAP[135] * dm)
pred90, pred135 = np.array(pred90), np.array(pred135)
Tl = [5, 10, 15]
out["V6"] = dict(T=Tl,
    chen_meas90=[L[L[:, 6] == t, 9].mean() for t in Tl], chen_meas135=[L[L[:, 6] == t, 10].mean() for t in Tl],
    law90=[pred90[L[:, 6] == t].mean() for t in Tl], law135=[pred135[L[:, 6] == t].mean() for t in Tl],
    wang_T=[11.8, 13.7, 15.7], wang={"90": [29, 21, 17], "60": [36, 34, 25], "30": [102, 90, 86]},
    wang_law=[(t / 13.7) ** (code_exp[4] - 1) for t in (11.8, 13.7, 15.7)], wang_fixed=[13.7 / t for t in (11.8, 13.7, 15.7)])
# V7: Wang edge error (H 10, c 10, F fixed from Chen baseline lag, T from WT codes)
F0 = 8 * 10 * 30.7e-6 / (2 * 0.045 - 0.025); V7 = []
for lab, T, th, m in (("Original, 90°", 13.734, 90, 18), ("Optimised, 90°", 15.696, 90, 11), ("Original, 60°", 13.734, 60, 30), ("Optimised, 60°", 15.696, 60, 24)):
    row = dict(label=lab, meas=m)
    for mode in ("pursuit", "discharge"):
        rr = g.run(g.Inputs(angle_deg=th, H_mm=10, c_mm=10, T_N=T, F_N=F0, corner_model=mode))
        row[mode] = dict(edge=max(um(s["edge"]) for s in rr.sections), e=um(rr.section("Mid-height")["error"]))
    V7.append(row)
out["V7"] = V7; print("V7", round(time.time() - t0))
json.dump(out, open(HERE / "paper_data.json", "w"), indent=1, default=float)
print("done", round(time.time() - t0))
