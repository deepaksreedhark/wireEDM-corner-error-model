"""Step 3 of 3: Word file with figures, captions and the validation table (needs python-docx).  python make_doc.py"""
import json, csv, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
d = json.load(open(HERE / "paper_data.json"))
mape = lambda m, p: 100 * np.mean(np.abs(np.array(p) / np.array(m) - 1))
# ---------------- validation metrics
V3a = d["V3a"]; V3b = d["V3b"]; V2 = {r["theta"]: r for r in d["V2"]}; O = d["V4"]["obara"]; Li = d["V4"]["liang"]; V5 = {r["theta"]: r for r in d["V5"]}
conf = {t: ([r["meas"][t] for r in V3b], [r["pred"][t] for r in V3b]) for t in ("45", "90", "135")}
T6 = d["V6"]; W = d["V7"]
rows = [
 ["Lag shape through the height", "Liang and Liao [8], 6 profiles", "Parabolic bow", "Profiles collapse onto the parabola", "Agrees", "Fig. 5"],
 ["Corner path error, 90°", "Chen et al. [3], baseline", "18.4 µm", f"{V2[90]['pursuit']['mid']:.1f} µm pursuit (+{100*(V2[90]['pursuit']['mid']/18.4-1):.0f} %); {V2[90]['discharge']['mid']:.1f} µm discharge (+{100*(V2[90]['discharge']['mid']/18.4-1):.0f} %)", "Pursuit within 20 %", "Fig. 6"],
 ["Corner path error, 135°", "Chen et al. [3], baseline", "8.6 µm", f"{V2[135]['pursuit']['mid']:.1f} µm pursuit (+{100*(V2[135]['pursuit']['mid']/8.6-1):.0f} %); {V2[135]['discharge']['mid']:.1f} µm discharge (+{100*(V2[135]['discharge']['mid']/8.6-1):.0f} %)", "Pursuit within 20 %", "Fig. 6"],
 ["Force law, 90° (fitted)", "Chen et al. [3], 27 L27 runs", "10–79 µm", f"leave-one-out mean error {mape(V3a['90']['meas'], V3a['90']['pred']):.0f} %", "Agrees", "Figs. 4, 7a"],
 ["Force law, 45° (predicted)", "Chen et al. [3], 27 L27 runs", "19–115 µm", f"mean error {mape(V3a['45']['meas'], V3a['45']['pred']):.0f} %", "Agrees", "Fig. 7a"],
 ["Force law, 135° (predicted)", "Chen et al. [3], 27 L27 runs", "8–63 µm", f"mean error {mape(V3a['135']['meas'], V3a['135']['pred']):.0f} %, biased low", "Under-predicts", "Fig. 7a"],
 ["Held-out runs, 90° / 45°", "Chen et al. [3], 8 confirmation runs", "24–79 µm", f"mean error {mape(*conf['90']):.0f} % / {mape(*conf['45']):.0f} %", "Agrees", "Fig. 7b"],
 ["Held-out runs, 135°", "Chen et al. [3], 8 confirmation runs", "18–32 µm", f"mean error {mape(*conf['135']):.0f} %, all low", "Under-predicts", "Fig. 7b"],
 ["Lag vs thickness (lower feed)", "Obara et al. [5], 20 / 80 mm", "113 / 242 µm", f"{O[0]['law']:.0f} / {O[2]['law']:.0f} µm (feed-only law: {O[0]['feedonly']:.0f} / {O[2]['feedonly']:.0f})", "Trend agrees", "Fig. 8a"],
 ["Bow vs thickness (lower feed)", "Liang and Liao [8], 50 / 70 mm", "112 / 135 µm (CPA)", f"{Li[1]['law']:.0f} / {Li[2]['law']:.0f} µm", "Agrees within 11 %", "Fig. 8b"],
 ["Spark-rate dip, 90° / 120°", "Hsue et al. [1]", "62 / 82 %", f"lagging wire {V5[90]['lag']:.0f} / {V5[120]['lag']:.0f} %; rigid wire {V5[90]['rigid']:.0f} / {V5[120]['rigid']:.0f} %", "Rigid limit agrees; lagging wire too high", "Fig. 9"],
 ["Tension trend, 5 → 15 N, 90°", "Chen et al. [3], main effects", f"{T6['chen_meas90'][0]:.1f} → {T6['chen_meas90'][2]:.1f} µm", f"{T6['law90'][0]:.1f} → {T6['law90'][2]:.1f} µm (fitted)", "Agrees (fitted)", "Fig. 10a"],
 ["Tension trend, second machine", "Wang et al. [4], 90°", "1.38 → 0.81", f"{T6['wang_law'][0]:.2f} → {T6['wang_law'][2]:.2f} (fixed F: {T6['wang_fixed'][0]:.2f} → {T6['wang_fixed'][2]:.2f})", "Does not transfer", "Fig. 10b"],
 ["Edge error at the part tip", "Wang et al. [4], 60° and 90°", "11–30 µm", f"{min(r['pursuit']['edge'] for r in W):.1f}–{max(r['discharge']['edge'] for r in W):.1f} µm", "Not captured", "Fig. 11"],
 ["Baseline cut with L27 force law", "Chen et al. [3], Table 2", "18.4 µm", f"{d['baseline']['e90']:.1f} µm (L27 run 5, same settings: {d['baseline']['run5']:.1f} µm)", "Data sets disagree", "—"],
]
hdr = ["Check", "Data", "Measured", "Model (v9)", "Outcome", "Figure"]
with open(HERE / "validation_summary.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(hdr); w.writerows(rows)
# ---------------- captions (citation numbers as in the course-project reference list)
CAP = [
 ("Fig01_lag_profile_and_section_paths", 6.5,
  "Static lag and corner paths for the default case (θ = 90°, H = 25 mm, guide clearance c = 10 mm, T = 9.8 N, F = 37 mN). (a) Wire lag y(z) between the guides from the tensioned-string model; markers show the five sections (faces, quarters, mid-height). (b) Plan view of the wire-centre paths at the faces, quarters and mid-height near the programmed corner C, with the corner path error e at each height (pursuit model)."),
 ("Fig02_kappa_vs_angle", 3.3,
  "Corner error coefficient κ = e/δ_mid against corner angle for the pursuit and discharge-circle force models (δ_mid = 30.7 µm, r_d = 150 µm). Stars: corner path error divided by the measured straight-cut lag from Chen et al. [3]."),
 ("Fig03_discharge_force_through_corner", 6.5,
  "Spark force through the corner from the discharge-circle model at 60°, 90° and 120° (δ_mid = 30.7 µm, r_d = 150 µm). (a) Force magnitude |f| and spark rate S/S₀ relative to a straight cut. (b) Rotation α of the force from the direction opposite to travel. Abscissa: guide position past the corner C in multiples of δ_mid."),
 ("Fig04_force_law_fits", 6.5,
  "Fitted terms of the process force law F = K (v_f·H/100)^a (t_on/12)^b (t_off/12)^c (V_p/40)^d (T/10)^g, from the 27 L27 runs of Chen et al. [3] at 90° (H = 25 mm). Each panel shows the force multiplier attributable to one setting (component-plus-residual), the level means (stars), the fitted term (solid) and ± one standard error of the exponent (dotted). K = 70.1 mN; R² = 0.961 on ln F."),
 ("Fig05_lag_shape_Liang", 3.3,
  "Normalised wire deflection through the workpiece height measured by Liang and Liao [8] for H = 30, 50 and 70 mm by camera picture analysis (CPA, filled) and cross-section measurement (CSM, open), against the parabolic bow of the string model with a uniform load."),
 ("Fig06_Chen_corner_error_vs_angle", 3.3,
  "Corner path error against corner angle for the baseline cut of Chen et al. [3] (25 mm SKD11, Ø0.25 mm brass), with the force set from the measured straight-cut lag. Lines: mid-height error of the pursuit and discharge-circle force models; bars: range from the faces to mid-height at the measured angles; stars: measured."),
 ("Fig07_force_law_validation_Chen", 6.5,
  "Corner path error predicted with the process force law and the pursuit model against measurements of Chen et al. [3]. (a) All 81 errors of the 27 L27 runs: 90° by leave-one-out refits, 45° and 135° predicted from the 90° fit. (b) The eight confirmation runs (Table 8 of [3]), not used in the fit. Percentages are mean absolute errors; the shaded band is ±20 %."),
 ("Fig08_thickness_validation", 6.5,
  "Effect of workpiece thickness when the feed is reduced for thicker parts. (a) Straight-cut lag of Obara et al. [5]; K set at H = 40 mm. (b) Maximum wire bow of Liang and Liao [8]; K set at H = 30 mm (CPA). Fixed force, a force law in feed only, and the removal-rate law of this work, F ∝ (v_f·H)^a. Feeds are given under each thickness."),
 ("Fig09_spark_rate_dip_Hsue", 3.3,
  "Minimum spark rate at the corner against corner angle: measurements and removal-rate model of Hsue et al. [1], the rigid-rod limit, and the discharge-circle model of this work for a rigid wire and for a wire lagging by 100 µm (r_d = 200 µm). Angles below 80° lie outside the scope of this study."),
 ("Fig10_tension_trend", 6.5,
  "Effect of wire tension. (a) Mean corner path error at 90° and 135° for each tension level of the L27 runs of Chen et al. [3], with the force law (tension term fitted to these data) and a fixed force (e ∝ 1/T). (b) Corner error relative to the value at 13.7 N on a second machine, Wang et al. [4], against the same two models."),
 ("Fig11_edge_error_Wang", 3.3,
  "Edge error at the part tip measured by Wang et al. [4] (10 mm SKD11; original and optimised settings) against the edge error of the pursuit and discharge-circle models; the discharge model's corner path error is shown for scale. Force set from the baseline lag of Chen et al. [3], since [4] reports no lag."),
]
doc = Document()
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(10)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
for s in ("Heading 1", "Heading 2", "Title"):
    doc.styles[s].font.name = "Times New Roman"; doc.styles[s].font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
doc.add_heading("Wire-EDM corner error modelling (v11): result and validation figures", level=1)
p = doc.add_paragraph(); p.add_run(
 "All model values were computed with simplified_corner_gui_v11.py (the same calculation as v9). Figures are provided as vector PDF and 600 dpi PNG in the paper_figures folder, "
 "sized for a single column (89 mm) or double column (184 mm). Reference numbers follow the reference list of the course project: "
 "[1] Hsue et al. 1999, [3] Chen et al. 2014, [4] Wang et al. 2022, [5] Obara et al. 2003, [8] Liang and Liao 2014. "
 "Renumber them if the list is reordered by first citation.").italic = True
doc.add_heading("Results", level=2)
for i, (fn, w_, cap) in enumerate(CAP, start=1):
    if i == 5: doc.add_heading("Validation", level=2)
    doc.add_picture(str(HERE / "figs" / f"{fn}.png"), width=Inches(w_)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = doc.add_paragraph(); r = c.add_run(f"Fig. {i}. "); r.bold = True; c.add_run(cap); c.paragraph_format.space_after = Pt(12)
doc.add_heading("Table 1. Validation summary", level=2)
t = doc.add_table(rows=1, cols=len(hdr)); t.style = "Table Grid"
for j, h in enumerate(hdr):
    cell = t.rows[0].cells[j]; cell.text = ""; run = cell.paragraphs[0].add_run(h); run.bold = True; run.font.size = Pt(8.5)
    sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:fill"), "E8ECF2"); cell._tc.get_or_add_tcPr().append(sh)
for row in rows:
    cells = t.add_row().cells
    for j, v in enumerate(row):
        cells[j].text = ""; run = cells[j].paragraphs[0].add_run(str(v)); run.font.size = Pt(8.5)
for j, wd in enumerate([1.25, 1.25, 0.85, 1.9, 0.95, 0.55]):
    for rr in t.rows: rr.cells[j].width = Inches(wd)
doc.add_paragraph()
doc.add_paragraph("Notes: 'Mean error' is the mean absolute percentage error. Chen et al. report tension in kgf; values are read as newtons. "
                  "The tension exponent of the force law is empirical and fitted to [3] only. Thickness checks lie outside the fitted feed range (3–5 mm/min).").runs[0].font.size = Pt(8.5)
sec = doc.sections[0]; sec.left_margin = sec.right_margin = Inches(0.9)
doc.save(HERE / "Paper_figures_and_validation.docx"); print("ok")
