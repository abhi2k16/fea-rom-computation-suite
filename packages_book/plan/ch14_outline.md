# Chapter 14 outline, replanned (awaiting approval)

Label `ch:verif`. Macro prefixes `cvf` (reproduction numbers), `cpl` (pipeline). Script `scripts/ch14_assets.py`, tables `tables/numbers_ch14.tex`.
Target: about 60 pages (now about 26). Sources: the 9 reproduction reports under `Physical_ML/ROM_formulations` (all read). Every number is a "report" number (from a development run) unless `ch14_assets.py` regenerates it; each study states which.

## Fixed template for each study section (4-6 pages)
1. **Concept and purpose**: what the published method does, which package modules carry it (`fea_engine` / `rom_engine` names), what the study tests.
2. **Mathematics as implemented**: governing equations, reduction, identification, integrator, error metric (numbered equations, the notation of the Notation section).
3. **Geometry and mesh**: dimensions, material, boundary conditions, element type, DOF count, one figure (redrawn mesh / geometry) and a **setup table** with a "paper / this work" column.
4. **Results**: 2-4 plots plus tables (modes, static fit, transient, backbone, ...).
5. **Comparison table against the publication**: quantity, paper, this work, verdict (reproduced / partly / not).
6. **Not reproduced and why**: deviations, unstated settings we chose, open defects.

## Section list
| § | Content | Pages |
|---|---|---|
| 14.1 | What "verified" means: six levels of evidence (kept), pyramid figure | 3 |
| 14.2 | Component verification table `tab:vf-map` (kept) | 3 |
| 14.3 | How the studies are reported: template above, verdict scale, register table `tab:vf-repro` (9 rows), conventions for "paper" vs "book" numbers, the same-author caveat | 2 |
| **Group A: structural nonlinear ROMs** | | |
| 14.4 | MFS-NLROM, He et al. 2023 | 6 |
| 14.5 | Hybrid ROM, Yang et al. 2019 (NonLin-HyROM) | 6 |
| 14.6 | ICE, Hollkamp and Gordon 2008 | 5 |
| **Group B: identification and learned models** | | |
| 14.7 | Loewner / FSI modal identification, Liu and Li 2026 | 5 |
| 14.8 | FD-PINN, Liang et al. 2024 | 5 |
| 14.9 | FRINO neural operator, Yang et al. 2025 | 5 |
| **Group C: contact and assemblies** | | |
| 14.10 | Hybrid hyper-reduced contact, Fauque et al. 2018 (1-D rope, 3-D pellet-cladding) | 6 |
| 14.11 | Projection-based contact ROM, Balajewicz et al. 2016 | 5 |
| 14.12 | Assembly MOR with contact and geometric nonlinearity, Nikiema et al. 2026 (2-D, 3-D joint) | 6 |
| 14.13 | Integrated parametric pipeline (old 14.4, kept) | 4 |
| 14.14 | Limits, summary, exercises (old 14.5, extended with a cross-study summary table) | 3 |

## Per-study content (from the reports)

### 14.4 MFS-NLROM (`Multi_Fidelity_NL_Structural_ROM`)
Concept: low-fidelity linear modes plus RBF surrogate of the nonlinear force, `q_nl = q_l - Lambda^-1 F_nl`; ICE baseline. Maths: governing equations, RBF kernels, OLHS training, Newmark with `newton` / `fixed_point` / `sign_deviation`, harmonic-balance backbone, R^2. Geometry: clamped-clamped flat beam; wing (shell reference, solid secondary because of the 78:1 aspect ratio). Results: static pressure sweep (paper Fig. 6, Table 2), free decay (Fig. 7), NNM backbone (Fig. 8); fidelity notes and the two bugs found. Figures: existing `results/` plots.

### 14.5 NonLin-HyROM (`NonLin-HyROM`)
Concept: modal ROM with F_nl identified from static solves (implicit condensation, quadratic + cubic Nash form). Maths: FOM, modal projection, F_nl identification, OLHS loads, Newmark, error metric, backbone test. Case 1 plate with hole (cantilever 1 x 0.2 x 0.01 m, 1224 DOF): modes against paper Table 1 (0.3 / 0.9 / 2.8 / 2.9 %), elastica benchmark of the curvature fix, FOM transient (12.8 % nonlinear gap), ROM 10.4 % against linear ROM 12.6 %, backbone over-hardening 4.7-4.9x (open defect D6). Case 2 cylindrical panel (R 2540, L 508, t 12.5 mm, 726 DOF): mesh figure, static response (softening then hardening), fit table, transient 2.8-6.5 % against projection floor, backbone within 0.09 %. Solver fixes list. Verdict: Case 2 reproduced, Case 1 partly.

### 14.6 ICE (`ICE-ROM`)
Concept: implicit condensation and expansion on the flat beam. Maths: Eqs. (1)-(13) of the report: modal equation, restricted cubic form (16 terms for 4 modes), least-squares identification, membrane expansion `T_m`, Newton-Newmark, fibre stress. Geometry: 18 x 1 x 0.09 in beam, 36 `Beam2DCorotational` elements. Results: mode table (within 0.21 %), static R^2 (0.9998 / 0.99935-0.99998), one-mode dynamics (19.5 % low, 112-136 % high) and the training-set / seed study, in-plane spectra (Fig. 1 not reproduced), four-mode variants and neural surrogate divergence, integrator equivalence, stress recovery. Verdict table (6 claims).

### 14.7 Loewner / FSI (`ROM_FluidStructur_Modal_Ident`)
Concept: matrix-free modal identification from scalar FRF samples. Maths: Loewner pencil, mode shapes, MAC, cross-ROM screening (Eqs. 21-32), Rayleigh-integral fluid loading, Newton on the nonlinear eigenproblem. Geometry: 455 x 379 x 3 mm steel plate, Mindlin 16 x 16 mesh (803 DOF), figures 00-02. Results S1-S7: Table 2 comparison, mesh convergence, boundary conditions, excitation cases, sensitivity (density, band, sensor, excitation), noise, MAC matrix. Not reproduced: ANSYS/BEM data, triangular shell variant, two-sided fluid.

### 14.8 FD-PINN (`FD-PINN_dynamics`)
Concept: moving-load beam solved per frequency. Maths: beam PDE, transform, real/imaginary residual split, loss, inverse EI, accuracy and nMSE definitions, band-truncation formula, f1 sensitivity. Geometry: simply supported beam L = 20 m, three speeds (table). Results: reference-solution agreement and band truncation table, forward Case 3 (time-domain PINN and FD-PINN), inverse accuracy table (100.0 / 98.9 / 94.6 %), Case 3 bias runs, explanation table with status labels. Needs PyTorch: numbers are report numbers, figures redrawn from saved `out_*.json` only if the files are in the folder (to be checked).

### 14.9 FRINO (`FE_ROM_Neural_Op`)
Concept: FNO constrained by a 3-mode POD Galerkin ROM residual. Maths: Hex20 FOM, POD, ROM, Rayleigh damping, FNO operator, physics loss, Eq. (20) frequency solve, error floor identity. Geometry: 60 x 15 x 1.5 mm cantilever, 14 x 8 x 2 Hex20, 4239 DOF (figure to draw). Results: FOM/POD/ROM validation table, 12-load error table against paper, forced-energy split (why the paper's 0.3 appears), training-set and seed study, superposition defect, ablations (lambda, width, order), PINN baselines, cost comparison. Remaining sections of the report (6-9) to be read when this study is written.

### 14.10 Hybrid hyper-reduced contact (`contact-hybrid-rom`)
Concept: displacements reduced, multipliers on the RID kept as FE unknowns. Maths: Signorini saddle point, POD / DEIM RID, hybrid reduced system, inf-sup repair, NNLS reconstruction, indicator eta, greedy, PB comparison, condensed LCP. Part I: 1-D rope (100 nodes), obstacle figure, basis-decay, solutions, error maps, Table 1 analogue, greedy convergence, two failure modes, RID-size finding (1 ring 0.49 %, 3 rings 0.011 %). Part II: 3-D pellet-cladding (1944 Hex8, 6937 DOF, 252 pairs), validation of the FE model, FOM fields, case table A1-B4 (training set rank-deficient), mesh study (3.2x / 11x / 50x), extrapolation. Missing-information table (13 items).

### 14.11 Projection-based contact ROM (`projection_MOR_contact`)
Concept: POD primal basis, NMF dual basis, greedy sampling with a solve-free indicator. Maths: Eqs. 3-15 of the paper as implemented, reduced LCP. Geometry: static obstacle membrane (200 x 200 Q4, 39,601 DOF), dynamic obstacle (1,521 DOF), two clamped Mindlin plates (3,174 DOF, 529 constraints). Results: building-block checks C1-C6, greedy against random sampling (5.0e-4 % against 6.3e-2 %), dynamic NNMF against SVD dual (0.08 % against 307 %), two-body time histories and down-sampling, package capability table. Open: full-size dynamic cases.

### 14.12 Assembly MOR (`MOR_struct_contact_geomNL`)
Concept: POD, DEIM, UDEIM, ECSW and domain partitioning for assemblies with contact and geometric nonlinearity. Maths: total-Lagrangian SVK element, node-to-segment penalty, Newton, mode criterion (Eq. 24), DEIM / UDEIM / ECSW formulas, DP condensation. Geometry: 2-D top-hat strip on flat strip (324 Quad4, 800 DOF), figures geometry and mesh; 3-D scaled butt joint (480 Hex20, 12,087 DOF). Results: element validation, FOM with stated data (contact too early), 28-case sensitivity, two candidates, mode counts, ROM study (Tables 6.1, 6.2), DP, Hex8 locking, preload independence and spring-connector fix, 3-D ROM accuracy and speed-ups, DEIM-NL variant (marked as own addition). Sections 11.3-14 to be re-read when written.

## Step plan after approval
1. **Group A** (14.4-14.6) with infrastructure: new `ch14_assets.py` functions, shared figure/table macros, template check. Confirm.
2. **Group B** (14.7-14.9). Confirm.
3. **Group C** (14.10-14.12). Confirm.
4. 14.1-14.3, 14.13-14.14 revisions, Appendix D entries, code-map rows, README, plan, abbreviations/notation additions (NMF, DEIM, UDEIM, RID, LCP, MAC, FNO, PINN), full rebuild.

## Constraints and risks
- Figures: the reports reference PNGs in the study folders. Two options per figure: copy the saved PNG into `packages_book/book/figures/ch14/` (fast, shows report plots) or redraw from saved `.npz/.json` (consistent style). Default: redraw where data are saved, copy otherwise; mesh/geometry figures drawn fresh in TikZ or matplotlib.
- Sandbox has no torch and 2 CPUs: FD-PINN and FRINO are report numbers only.
- Same author as the software; unstated settings were chosen by us; failures are reported as such.
- No design-lesson arguments: sections state what was done, what was measured and what is open.
- Page count may reach 65-70 if figure-heavy; will report after Group A.

## Register of reports
| # | Study | Report | Group |
|---|---|---|---|
| 1 | MFS-NLROM, He 2023 | `Multi_Fidelity_NL_Structural_ROM/MFS_NLROM_Reproduction_Report.md` | A |
| 2 | HyROM, Yang 2019 | `NonLin-HyROM/NonLin-HyROM_Reproduction_and_Validation.md` | A |
| 3 | ICE, Hollkamp 2008 | `ICE-ROM/ICE_ROM_reproduction_results.md` | A |
| 4 | Loewner/FSI, Liu 2026 | `ROM_FluidStructur_Modal_Ident/ROM_reproduction_validation_report.md` | B |
| 5 | FD-PINN, Liang 2024 | `FD-PINN_dynamics/FD-PINN Reproduction Validation Against Liang et al. (2024).md` | B |
| 6 | FRINO, Yang 2025 | `FE_ROM_Neural_Op/FRINO_reproduction_results.md` (+ `FRINO_deep_analysis.md`) | B |
| 7 | Hybrid contact, Fauque 2018 | `contact-hybrid-rom/Hybrid_HROM_Contact_Mechanics_Reproduction_Report.md` | C |
| 8 | Projection contact, Balajewicz 2016 | `projection_MOR_contact/Contact_ROM_Validation_Report.md` | C |
| 9 | Assembly MOR, Nikiema 2026 | `MOR_struct_contact_geomNL/Nikiema2026_assembly_MOR_reproduction_report.md` | C |

Excluded: nonlinear rods POD/SIM (no report), `Efficient_ROM_AC` (PDF only).

## Speed-up discrepancy (carried over)
README: 221.8x; `CLAIMS.md` E1: about 260-270x; `paper/results/e1_rom_speedup.json`: 361.2x (author's 12-CPU machine, commit 33cde35); 254x appears nowhere. Chapter 14 quotes the book's regenerated ratio plus 221.8x and 361x, each with its source. Release tag `release-fea1.0.1-rom0.2.0` at f87baf3 (local, not pushed).
