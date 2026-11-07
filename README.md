# Wire-EDM corner error modelling — final script package (v11, 1 Nov 2026)

<img width="955" height="590" alt="image" src="https://github.com/user-attachments/assets/2fca3390-95b0-47cc-acf9-26717d60dcde" />
<img width="950" height="593" alt="image" src="https://github.com/user-attachments/assets/ada745bd-43b2-4f8a-9c80-e4013f52c628" />

## Run the tool
```
python simplified_corner_gui_v11.py
```
Keep `plotly.min.js` in the same folder (the browser 3-D view embeds it, so it works offline).
Needs Python 3.10+ with numpy, matplotlib and tkinter (included in the python.org Windows installer). The browser view also uses contourpy (installed with matplotlib).

## Contents
| File | What it is |
|---|---|
| `simplified_corner_gui_v11.py` | The tool: calculation, plots, browser view and window in one file |
| `plotly.min.js` | Plotly.js, embedded into the browser view |
| `Simplified_corner_model_methodology_rev6.docx` | Method, formulae, dimensioned diagrams, validation and sources (describes the v9 calculation, unchanged in v10–v11) |
| `reproduce_paper_figures/` | Scripts that regenerate the 11 paper figures, captions and validation table |

## What the tool calculates
1. **Spark force:** fixed F, or the process law `F = K (v_f·H/100)^1.456 (t_on/12)^1.258 (t_off/12)^-0.203 (V_p/40)^0.746 (T/10)^0.843`, fitted to Chen et al. 2014 (K = 0.0701 N; *Set K from this lag* recalibrates K for another machine from one measured straight-cut lag).
2. **Static lag:** tensioned string, `δ_mid = F(2L − H)/8T`.
3. **Corner path:** pursuit (default) or the discharge-circle force balance.
4. **Five sections:** one-body wire, corner path error at faces, quarters and mid-height.
5. **Machined part:** slot swept by the discharge circles; edge error at the tip; interactive 3-D view.

<img width="1626" height="834" alt="image" src="https://github.com/user-attachments/assets/0179cf80-ea1f-4666-9b52-f9bf4b446199" />


Valid ranges and limitations are in the methodology document (Sections 7 and 11).

## Reproduce the paper figures
From `reproduce_paper_figures/`:
```
python compute.py      # about 10 minutes; writes paper_data.json
python figures.py      # writes figs/*.pdf and figs/*.png
python make_doc.py     # writes Paper_figures_and_validation.docx and validation_summary.csv (needs python-docx)
```
`data/` holds the measured literature values used for validation (collated from WEDM_model_validation_data_v2.xlsx).
