# Changelog: simplified corner-error model

From v3 on, each version is **one file**, `simplified_corner_gui_vN.py`, holding the calculation, plots, browser view and window.
Every change is saved as the next version (v4, v5, …). Earlier versions are never edited.
`plotly.min.js` must sit in the same folder; all versions share it.
`Simplified_corner_model_methodology_revN.docx` is revised with every version, each revision saved as a new file (rev1 and rev2 describe v5, rev3 describes v6, rev4 describes v7, rev5 describes v8, rev6 describes v9). Superseded revisions move to `archive/`.

v1 and v2 used separate model, gui and html files. They and the pre-v3 working copies are in `archive/`. Older single-file versions are moved there too as new ones replace them.

## v11 (1 Nov 2026): `simplified_corner_gui_v11.py`
- Window and browser-view title: "Wire-EDM corner error modelling v11 1 Nov 2026". No change to the calculation.
- Final script package saved to `Research_paper/Final_script_package`.

## v10 (1 Nov 2026): `simplified_corner_gui_v10.py`
- Fix: *Set K from this lag* stopped with "could not convert string to float" when the lag box was empty, used a comma decimal (`30,7`) or had a unit typed after the number (`30.7 µm`). All number boxes now accept a comma decimal and ignore a trailing unit; an empty lag box gives a message saying what to enter; a wrong entry names the box.
- No change to the calculation; methodology stays at revision 6.
- Browser view saves as `corner_view_v10.html`.

## v9 (1 Nov 2026): `simplified_corner_gui_v9.py`
- Process force law in removal-rate form: `F = K (v_f·H/100)^1.456 (t_on/12)^1.258 (t_off/12)^-0.203 (V_p/40)^0.746 (T/10)^0.843`. The feed term now uses feed × thickness, so a thicker wall at the same feed gives more force.
- Identical to v7–v8 at Chen's 25 mm (all L27 and confirmation-run results unchanged).
- Fixes the thickness trend where thicker parts are cut at lower feed: Obara 2003 lag 112 / 201 µm vs 113 / 242 measured (v8: 307 / 73); Liang & Liao 2014 bow 100 / 139 µm vs 112 / 135 (v8: 47 / 41).
- The results table shows the removal rate in law mode. Browser view saves as `corner_view_v9.html`.
- Methodology revision 6: Section 7.1 rewritten, new Section 7.6 (thickness test), Figure 5 updated.

## v8 (1 Nov 2026): `simplified_corner_gui_v8.py`
- Fix: the discharge-circle model gave an erratic wire path and force spikes below 90°. Cause: the on/off uncut test made the force jump when a circle point crossed the slot edge, so the balance did not converge (31–35 of the first 40 steps after the corner at 45–75°, residuals up to 29 µm).
- The uncut test is now a smooth 3 µm contact zone centred on the slot edge, U = clamp(0.5 + d/ε, 0, 1); solver damping 0.2, up to 150 iterations. Every step converges at 30–150°.
- Results at 90°: e = 24.3 µm (was 24.7), edge 2.4 µm; 135°: 10.3 µm (was 10.5). Acute corners now valid: κ = 0.965 at 45°, 0.946 at 60°.
- Pursuit mode and the force law are unchanged. Browser view saves as `corner_view_v8.html`.
- Methodology revision 5 (Sections 6.1, 6.3, 6.6, 6.7 updated). Before/after comparison: `validation/v8_acute_fix.png`.

## v7 (1 Nov 2026): `simplified_corner_gui_v7.py`
- New optional **process force law** (*Spark force → Process law*): `F = K (v_f/4)^1.456 (t_on/12)^1.258 (t_off/12)^-0.203 (V_p/40)^0.746 (T/10)^0.843`.
    - Exponents fitted to the corner error at 90° in Chen 2014's L27 experiment (27 runs) through the pursuit model. R² 0.96; leave-one-out error 13 % (48 % without the tension term).
    - Held-out test on Chen's 8 confirmation runs: mean error 11 % at 90°, 15 % at 45°; 135° under-predicted by 37 % (the obtuse excess also seen in v6 validation).
    - K = 0.0701 N (Chen's L27 machine). *Set K from this lag* fits K to one measured straight-cut lag at the current settings.
    - The tension exponent is empirical; the law is valid only inside the fitted ranges.
- New inputs: feed, pulse-on, pulse-off, peak voltage, K, measured lag. The F entry is disabled in law mode. The results table shows the force used.
- Browser view saves as `corner_view_v7.html`.
- Methodology revision 4: `Simplified_corner_model_methodology_rev4.docx` (new Section 7, Figure 5).
- Validation of the law in `validation/plots_v7/` (`fit_force.py`, `plot_force.py`).

## v6 (1 Nov 2026): `simplified_corner_gui_v6.py`
- New optional corner model, **discharge-circle force**, chosen with the *Corner model* buttons (pursuit stays the default):
    - Spark weight `w = max(0, n·v)` on the points of the discharge circle that are still uncut; v is the pursuit line from the previous wire position to the guide. The last 0.2 δ_mid of the wire's own track counts as the cutting front, not old slot.
    - Spark rate `S = S_area^(1−c)`, force `f = m·f_area` with `m = min(S_area^(−c), 5)`; force balance `W = G + δ_mid·f`, solved by damped iteration at steps of δ_mid/20.
    - New inputs: discharge concentration c (0–1) and gap growth k_g [µm], `r = r_d + k_g(m − 1)`. Both default to 0.
    - New plot panel: force size, spark rate and force rotation along the path.
- **Edge error** (Wang 2022) at the five heights for both models: material removed from the ideal tip P along the bisector.
- The results table scrolls and shows lag | error | edge per section, plus the discharge summary rows.
- Browser view: the part is coloured by material removed beyond the ideal edge (horizontal colour bar); the slot uses each wire position's own radius. Saves as `corner_view_v6.html` (about 14 s to build in discharge mode).
- Results at 90°, defaults: pursuit e = 20.3 µm, edge 0.25 µm; discharge e = 24.7 µm, edge 2.65 µm, min S/S₀ 0.94, force rotation ≤ 1.1°.
- Methodology revision 3: `Simplified_corner_model_methodology_rev3.docx` (new Section 6 and Figure 4).

## v5 (31 Oct 2026): `simplified_corner_gui_v5.py`
- The browser view's machined block is now a closed triangle mesh (`Mesh3d`) instead of a Plotly isosurface, whose end caps rendered wrongly.
- How the mesh is built:
    - At 9 heights, the cross-section outline of the part and of the scrap side is traced with marching squares (contourpy), interpolated between grid points.
    - Each outline is resampled to 480 points from the same corner, and side walls are lofted between heights.
    - The top and bottom caps are ear-clipped polygons.
- The grid is finer (221 × 221, about 4 µm). The slot distance is exact point-to-segment on the straight legs and sampled at about 1 µm round the corner, so building the view takes about 6 s.
- The browser view saves as `corner_view_v5.html`.
- New methodology document, revision 1: `Simplified_corner_model_methodology.docx` covers v5.

## v4 (31 Oct 2026): `simplified_corner_gui_v4.py`
- The matplotlib 3D view keeps the orientation you set (elevation, azimuth, roll) whenever it redraws.
- Moving the guide slider also keeps your zoom and pan. After **Calculate**, the zoom refits to the new result but the orientation stays.
- The browser view saves as `corner_view_v4.html`. Its own slider already kept the camera.

## v3 (31 Oct 2026): `simplified_corner_gui_v3.py` (single file)
- Times New Roman everywhere: the window, the matplotlib plots (falls back to another serif font if missing) and the browser view.
- Text scales with the screen:
    - Window: DPI-aware on Windows. Font points follow the DPI and grow by up to 1.8× on screens taller than 1080 logical pixels. The window opens at 90 % of the screen.
    - Plots: matplotlib text sizes are now relative to one base size, which scales the same way.
    - Browser view: fonts scale with the window height and are recomputed on resize, keeping the current camera.
- Browser view:
    - The machined part is fully opaque. The scrap side is hidden by default; buttons give Part only, Part + scrap side (solid) or Scrap see-through.
    - New slider moves the guides and wire along the path (about ±400 µm round C, 42 positions).
- The GUI slider redraws on its own as it moves; the Redraw button is gone.
- The block's distance field is clipped near the surface, which removes the stray flanges Plotly drew on slanted walls (seen at 120°).
- The calculation, plots, browser view and window are merged into one file. The browser view saves as `corner_view_v3.html`.

## v2 (31 Oct 2026): `simplified_corner_model_v2.py`, `simplified_corner_gui_v2.py`, `simplified_corner_html_v2.py`
- Workpiece faces in the matplotlib 3D view are more opaque (alpha 0.10 → 0.4).
- New `simplified_corner_html.py`: an interactive Plotly view in the browser. It shows the machined block (solid part, translucent scrap side, slot as the envelope of the discharge circles), the wire, the guides and the paths at the five heights, plus the lag-profile and top-view panels.
- The GUI has a button that writes the HTML view (`corner_view_v2.html` for v2) and opens it. `plotly.min.js` sits in the folder so the HTML works offline.

## v1 (31 Oct 2026): `simplified_corner_model_v1.py`, `simplified_corner_gui_v1.py`
- Two steps: the mean lag from a string model, then the corner pursuit (tractrix) driven by the mid-height lag.
- The whole wire moves as one body (Step 11), so the corner path error is reported at five heights: top face, upper quarter, mid-height, lower quarter and bottom face.
- Single default force, F = 0.037 N (Chen 2014, 30.7 µm lag). There are no presets and no force options.
- Plots:
    - 3D view: workpiece slab, guides, deflected wire and the paths at the five heights.
    - Lag through the height.
    - Top view of the corner.
- GUI has seven inputs, a results table and a guide-position slider.
- Defaults (90°, H = 25 mm, c = 10 mm, T = 9.8 N): δ_mid = 30.7 µm, δ_face = 18.9 µm, e = 20.3 µm at mid-height and 17.9 µm at the faces.

An earlier draft (presets, three force options, validation table) was replaced by v1 the same evening and is not kept.
