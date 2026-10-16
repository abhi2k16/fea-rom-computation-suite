<div align="center">

# computation-suite

**From-scratch finite element analysis and reduced-order modeling in Python**

![fea_engine](https://img.shields.io/badge/fea__engine-1.0.1-2563eb?style=for-the-badge)
![rom_engine](https://img.shields.io/badge/rom__engine-0.2.0-7c3aed?style=for-the-badge)
![python](https://img.shields.io/badge/python-3.9+-3776ab?style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-1.22+-013243?style=for-the-badge)
![SciPy](https://img.shields.io/badge/SciPy-1.8+-0054a6?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-optional-ee4c2c?style=for-the-badge)
![license](https://img.shields.io/badge/license-MIT-16a34a?style=for-the-badge)

[**🔗 github.com/abhi2k16/fea-rom-computation-suite**](https://github.com/abhi2k16/fea-rom-computation-suite) &nbsp;|&nbsp; [⬇️ Clone](#s-2) &nbsp;|&nbsp; [🚀 Quick start](#s-3)

</div>

## 🧭 Contents

| | |
|---|---|
| 📦 [Packages](#s-1) | 📚 [Documentation](#s-7) |
| ⚙️ [Install](#s-2) | 🗂️ [Repository layout](#s-8) |
| 🚀 [Quick start](#s-3) | ✅ [Tests](#s-9) |
| 🔄 [A typical workflow](#s-4) | ⚠️ [Notes and limits](#s-10) |
| 🖥️ [CPU, multi-core and GPU](#s-5) | 📄 [License](#s-11) |
| 🧪 [Examples](#s-6) | 🎨 [At a glance](#glance) |
| 🧲 [Contact module](#contact) | |
| 🖼️ [Example gallery](#gallery) | |
| ⚡ [Important note: batched kernels](#batched) | |

---


A from-scratch, NumPy/SciPy-native finite element and reduced-order modeling
toolkit for structural mechanics, with an optional PyTorch layer for GPU
acceleration and automatic differentiation.

Every result is checked against an independent reference: a closed-form
solution, a second numerical method, or a conservation law. Nothing here is
a wrapper around a commercial solver.

<a id="glance"></a>

## 🎨 At a glance

### Workflow overview

<div align="center">
<img src="docs/images/architecture.png" alt="How fea_engine and rom_engine fit together: model definition, FESystem, analyses and results in fea_engine; snapshots, reduced basis, reduced models and fast online use in rom_engine; K, M, C, F passed as plain arrays; optional PyTorch" width="900">
</div>

`fea_engine` turns a model definition into assembled matrices through `FESystem` and runs the analyses. Its only output to `rom_engine` is a set of plain arrays `K, M, C, F`, so `rom_engine` also accepts matrices from any other source. `rom_engine` builds a basis from full-order snapshots once (offline) and then evaluates reduced models per query (online). The dashed arrow is the check of a reduced model against the full model on unseen inputs (`validate_rom`). PyTorch is optional and used by both packages. This is the same figure as Figure 1 of the accompanying paper; its LaTeX source is `docs/images/architecture.tex`.

### Package map


```mermaid
flowchart TB
    subgraph FE["🏗️ fea_engine - full-order model"]
        direction TB
        MAT["Material<br/>Section"] --> SYS
        MSH["Mesh<br/>Geometry"] --> SYS
        ELM["Elements<br/>truss · beam · solid<br/>plate · shell · contact"] --> SYS
        SYS["<b>FESystem</b><br/>assemble K, M, F<br/>loads · constraints"]
        SYS --> A1["Static"]
        SYS --> A2["Modal<br/>Buckling"]
        SYS --> A3["Harmonic<br/>Random vibration"]
        SYS --> A4["Transient"]
        SYS --> A5["Nonlinear<br/>Newton · arc-length<br/>plasticity · contact"]
    end
    subgraph RE["📉 rom_engine - reduced model"]
        direction TB
        POD["POD basis"] --> PRJ["Galerkin<br/>Affine sweeps<br/>Frequency ROM"]
        SSM["State-space MOR<br/>balanced truncation<br/>Krylov · SOAR · Hankel"]
        NLR["Nonlinear ROMs<br/>RBF · ICE · intrusive · NNM"]
        IDN["Loewner<br/>identification"]
    end
    SYS -- "snapshots" --> POD
    SYS -- "K, M, C" --> SSM
    SYS -- "K, M, F" --> NLR
    TCH[("🔥 PyTorch (optional)<br/>CPU · GPU · autograd")]
    FE -.-> TCH
    RE -.-> TCH
    classDef fea fill:#2563eb,stroke:#1e40af,color:#fff
    classDef rom fill:#7c3aed,stroke:#5b21b6,color:#fff
    classDef out fill:#0f766e,stroke:#115e59,color:#fff
    classDef torch fill:#ee4c2c,stroke:#b91c1c,color:#fff
    class SYS,MAT,MSH,ELM fea
    class POD,PRJ,SSM,NLR,IDN rom
    class A1,A2,A3,A4,A5 out
    class TCH torch
```

### What is inside

```mermaid
mindmap
  root((computation-suite))
    fea_engine
      Elements
        Truss, Beam, Solid
        Plate, Shell, Contact
      Analyses
        Static, Modal, Buckling
        Harmonic, Random vibration
        Transient
      Nonlinear
        Large displacement
        Plasticity, Hyperelasticity
        Arc-length, Contact
      Tools
        Meshing and grading
        Adaptive refinement
        Topology optimisation
    rom_engine
      Projection
        POD, Galerkin, Affine
      Frequency domain
        Greedy training
        Certified error bounds
      State space
        Balanced truncation
        Krylov, SOAR, Hankel
      Nonlinear
        RBF, ICE, Intrusive
        Normal modes
      Identification
        Loewner
    PyTorch layer
      GPU solves
      Autograd tangents
      Neural operators
```

### Why reduce a model: offline once, online many times

```mermaid
flowchart LR
    subgraph OFF["🧮 Offline - once, expensive"]
        direction LR
        a["Full-order solves<br/>at sampled loads / parameters"] --> b["Snapshot matrix"] --> c["POD basis V"] --> d["Project K, M, F<br/>into small operators"]
    end
    subgraph ON["⚡ Online - many times, cheap"]
        direction LR
        e["New load or<br/>parameter value"] --> f["Reassemble<br/>tiny matrix"] --> g["Solve r x r<br/>system"] --> h["Expand<br/>x = V q"]
    end
    d --> f
    classDef off fill:#b45309,stroke:#92400e,color:#fff
    classDef on fill:#15803d,stroke:#166534,color:#fff
    class a,b,c,d off
    class e,f,g,h on
```

Measured on the two-region beam example in `USER_GUIDE.md` (120 elements, 500-point parameter sweep, relative error about 1.5e-3):

```text
Full-order   1653.23 ms  ██████████████████████████████████████████████████████████████████████
Reduced-order   7.45 ms  ▏                                                         221.8x faster
```

The example gallery, including the contact examples, follows the contact module section below.

<a id="s-1"></a>

## 📦 Packages

| Package | Purpose |
|---|---|
| [`fea_engine`](fea_engine/README.md) | Full-order finite element package: static, dynamic and geometrically nonlinear analysis, with extensible materials, elements, loads and solvers. |
| [`rom_engine`](rom_engine/README.md) | Reduced-order modeling built on top of `fea_engine`. |

The two work together: build K, M and F with `fea_engine`, then hand those
plain arrays to `rom_engine`. `rom_engine` needs nothing from `fea_engine` at
run time (the `fea` extra only installs it for the examples and tests).

### What `fea_engine` covers

- **Elements:** trusses (linear, large-displacement, plastic); 2-D and 3-D beams (Euler-Bernoulli, corotational, geometrically exact Reissner); plane stress/strain triangles and quadrilaterals (linear and quadratic); 3-D solids (hexahedra and tetrahedra, linear and quadratic, B-bar); Mindlin plates; MITC shells (linear, corotational, director-based); contact elements (gap, node-to-segment, with friction).
- **Analyses:** static, linear buckling, modal, harmonic / frequency sweep, random vibration (PSD), implicit and explicit transient, nonlinear static with load stepping, displacement control, arc-length and Koiter-Newton, nonlinear transient (Newmark-Newton).
- **Nonlinearity:** large displacement and rotation, J2 plasticity (isotropic and kinematic hardening), Neo-Hookean hyperelasticity, penalty and Lagrange-multiplier contact.
- **Mesh and loads:** structured mesh generators (lines, rectangles, boxes, with holes and notches), graded meshes, multi-block meshes, mesh file I/O (`meshio`), 3-D contour, deformed-shape, animation, slice and full-vs-reduced comparison plots (`pyvista`, optional), edge and face pressure loads by quadrature, adaptive refinement with error estimators.
- **Performance and GPU:** opt-in batched nonlinear element kernels (Quad4/Hex8/Hex20/Tet10 total-Lagrangian), sparse Newton path, batched contact sets and hyper-reduction hooks (`fea_engine/docs/batched_nonlinear_kernels_plan.md`), vectorised assembly, sparse and iterative solvers (CG with Jacobi/SSOR preconditioning), batched multi-load solves, torch CPU/GPU solve backends, autograd tangents, topology optimisation.

### What `rom_engine` covers

- **Projection ROMs:** POD basis (optionally mass-weighted) and Galerkin reduction for static, modal and transient problems; affine parametric decomposition for fast parameter sweeps.
- **Frequency-domain ROMs:** modal/POD/greedy bases, frequency response, random-vibration response, and certified error bounds (SCM).
- **Systems-and-control reduction:** balanced truncation (plain and frequency-weighted), singular perturbation, optimal Hankel-norm approximation, Krylov and second-order (SOAR) moment matching, passivity checks.
- **Non-intrusive identification:** Loewner-pencil modal identification with cross-ROM stability screening.
- **Nonlinear structural ROMs:** RBF-surrogate and polynomial (ICE-style) models, intrusive projection of nonlinear internal forces, membrane expansion, nonlinear normal mode backbone curves.
- **Training tools:** optimal Latin hypercube sampling, dataset coverage diagnostics, comparison metrics.
- **Research-grade (PyTorch):** neural operators, parameterised latent ODEs, reduced-basis operators, differentiable correction, ensemble uncertainty. Treat these as prototypes.

<a id="contact"></a>

## 🧲 Contact module

Unilateral (frictionless) contact is solved exactly with Lagrange multipliers, in both packages, on plain arrays and without an element class:

```
K u = f + Bᵀλ,    B u − c ≥ 0,    λ ≥ 0,    λ·(B u − c) = 0
```

`fea_engine` solves the full-order problem; `rom_engine` reduces it (a POD basis for the displacements, a non-negative basis for the contact forces) and never imports `fea_engine`. Penalty contact elements, which came earlier, are listed for completeness. **Bold** marks what was added by the contact work of 2026-10-16 (Wave 23).

### `fea_engine`: contact modules, elements and methods

| Module | What it provides | Status |
|---|---|---|
| `contact_solvers.py` | `solve_contact_static` (dense: `active_set`, `lemke`, `pgs`, and now `pdas`), `solve_contact_newton`, `solve_contact_dynamics` | earlier waves; extended below |
| `contact_solvers.py` | **`solve_contact_static_sparse(K, f, B, c, route="auto")`**: primal-dual active set with the routes `reduced`, `saddle`, `condensed`, warm start `active0`, opt-in `fallback=True`, `max_dense_bytes` guard | **new** |
| `contact_solvers.py` | **`solve_contact_dynamics(..., scheme="bdf2", return_info=True)`** and the record **`ContactDynamicsInfo`** (scheme, ok, failed_step, reason, message, suggestion, V, iterations) | **new** (default `two_step` unchanged) |
| `elements/scalar.py` | **`Quad4Laplace`** (registry key `quad4_laplace`): 4-node scalar-field element, one DOF `"u"` per node; stiffness, consistent and lumped mass, source vector | **new** element |
| `elements/contact.py` | `GapContactPenalty`, `GapContactCurvedFriction`, `NodeToSegmentContact2D`, `NodeToSegmentContact2DFriction`, `find_contact_pairs_2d` | earlier waves |
| `contact_batch.py` | `ContactSet2D`, `ContactSet3D`, `candidate_pairs_2d/3d`: penalty contact for many pairs at once | Wave 22 |
| `nonlinear_solver.py` | `solve_contact_lagrange_static` (one node, exact) | earlier waves |

**Changes to existing code (all opt-in; a call that worked before gives the same result):**

| Where | Change |
|---|---|
| `solve_contact_static(..., method="pdas")` | new method value; it calls `solve_contact_static_sparse` |
| `ContactResult` | new fields `route`, `reason`, `message`, `suggestion`, `active_mask`, `fallbacks` and the alias `ok`; a failing route says why and which route to try next |
| `solve_contact_dynamics` | new keywords `scheme` (`"two_step"` default, `"bdf2"`), `return_info`, `route`, `fallback`; `method="pdas"` for BDF2 only |
| `FESystem` assembly | no change in this work. The assembly-side contact changes are from Wave 22: `add_contact_set`, the cached sparse CSR tangent, `eliminate_constraints`. `Quad4Laplace` works with the unchanged `assemble_stiffness`, `assemble_mass`, `assemble_lumped_mass`, `fix_dofs`, `solve_modal` |
| package exports | `solve_contact_static_sparse`, `ContactDynamicsInfo`, `Quad4Laplace` exported from `fea_engine` |

### `rom_engine`: contact modules and methods

| Module | What it provides | Status |
|---|---|---|
| `contact_dual_basis.py` | `ContactROM`, `nonneg_dual_basis`, `active_snapshot_columns`; **`ContactROM(..., M=)`** and **`ContactROM.solve_dynamic(force, c, dt, n_steps, scheme="bdf2")`** returning **`ContactROMDynamicResult`** (reduced matrices factored once, warm-started LCP, a KKT guard that reports wrong LCP solutions) | static: earlier; **dynamic: new** |
| `contact_greedy.py` | `greedy_contact_sampling`, `contact_condition_indicator` (the reference); **trajectory snapshots**, **`rel_tol`**, **`pass_round`**, **`trajectory_indicator`**, **`linear_basis_schedule`**, **`ContactConditionIndicator`** (cached pseudo-inverse; skips the least-squares step for `B = I`) | extended / **new** |
| `complementarity.py` | `lcp_lemke`, `lcp_active_set`, `solve_lcp`, `solve_reduced_contact`; **`active0=`** warm start on `lcp_active_set` and `solve_lcp` | extended |
| `contact_hyper_reduction.py`, `contact_rid.py`, `domain_partition.py` | contact-aware ECSW, hybrid hyper-reduction on a reduced integration domain, linear/nonlinear partition with static condensation | earlier waves |
| `contact_comparison.py`, `contact_validation.py` | compare reduced contact models; error metrics | earlier waves |
| `friction_contact_models.py`, `torch_contact_bases.py` | dry-friction elements for harmonic balance; PyTorch counterparts of the dense contact bases | earlier waves |

**Changes to existing code (all opt-in):** `ContactROM` takes an optional mass matrix `M` and keeps `M_r`; `lcp_active_set` and `solve_lcp` take `active0`; `greedy_contact_sampling` accepts trajectory-valued snapshots and the keywords `rel_tol` and `pass_round`, and a call without them behaves as before. The `rel_tol` rule is the author's reading of line 15 of Algorithm 2 of Balajewicz, Amsallem and Farhat, which is printed with `>` and is probably a typo.

### What was measured

| Claim | Measured | Where |
|---|---|---|
| The sparse routes agree with the dense solvers | differences 5.4e-16 to 1.3e-15 on the 361 to 3,481-unknown membrane obstacle problems of the gallery; 7.4e-15 between the two sparse routes at 39,601 unknowns | `fea_09`; `fea_engine/tests/sparse_contact_timing.py` |
| The sparse routes scale | 39,601 unknowns, 9,425 active constraints, 30 iterations: `reduced` 4.5 s, `saddle` 7.6 s (12-CPU Windows machine); the dense route was not run (it needs about 25 GB) | `sparse_contact_timing.py` |
| Sparse beats dense where both fit | 3,481 unknowns: `reduced` 0.17 s, `saddle` 0.22 s, dense 3.97 s (sandbox, 2 CPUs) | `fea_09` figure |
| BDF2 is second order | free-membrane error at t = 2: 1.15e-2 / 2.96e-3 / 7.2e-4 for dt = 0.01 / 0.005 / 0.0025, observed order 1.96 and 2.03; `two_step` 0.162 / 0.0859 / 0.0454, order 0.92 | `fea_10` |
| BDF2 is not contact-stabilised | bar impact, h = 0.002: energy rises to 1.0060 of the initial value (up to 0.72% in the test suite), ends at 0.976; `two_step` never exceeds 1.000 and ends at 0.938. The BDF2 wall force shows spikes during the impact | `fea_10` |
| Complete bases reproduce the full order | reduced BDF2 trajectory equals the full-order one to 3.3e-11 | `rom_engine/tests/test_contact_dynamic_rom.py` |
| The reduced dynamic model is accurate | 324 DOFs, 120 steps, up to 54 active constraints: relative error 0.81% / 0.20% / 0.023% for p = p_λ = 10 / 30 / 40; reduced contact conditions hold (KKT residual at most 2.7e-18) | `rom_07` |
| Greedy sampling beats random sampling (static) | mean error on six held-out obstacles with 2 / 4 / 6 / 8 snapshots: greedy 4.75 / 1.55 / 1.23 / 0.82%, random median 7.37 / 2.94 / 2.11 / 1.63% | `rom_07` |
| The cached indicator is cheap | 0.04 to 0.45 ms per call up to 40,000 constraints; the reference function took 3.3 s at 2,500 | `rom_engine/tests/contact_indicator_timing.py` |

Times come from one run on one machine and are not asserted by any test; the gallery figures are the sandbox run named in each figure. Not covered: friction in the new solvers, Newmark contact dynamics, parametric prediction of the dynamic ROM, a dense-route run at 39,601 unknowns. Convergence of the active-set iteration is not guaranteed when `K` is not an M-matrix. Record: `fea_engine/docs/consolidated_future_roadmap.md`, Wave 23.

### Tests

| File | Tests | Covers |
|---|---|---|
| `fea_engine/tests/test_contact_solvers_sparse.py` | 28 | routes against dense solvers on six obstacles, failure reports, fallback, warm start, memory guard |
| `fea_engine/tests/test_quad4_laplace.py` | 17 | element matrices, patch test on distorted meshes, Poisson convergence, membrane eigenvalues |
| `fea_engine/tests/test_contact_dynamics_bdf2.py` | 14 | order of accuracy, contact conditions at every step, impact, input checks |
| `rom_engine/tests/test_contact_dynamic_rom.py` | 13 | complete bases against full order, KKT guard, warm start, POD + NMF accuracy |
| `rom_engine/tests/test_contact_greedy.py` | 13 | greedy selection (three of the 13 predate this work), trajectories, `rel_tol`, `pass_round` |
| `rom_engine/tests/test_contact_indicator.py` | 37 | `ContactConditionIndicator` routes against the reference |

All of these passed on the author's machine. In the full-suite runs there, `fea_engine` had 0 failures; `rom_engine` had the known operator-inference failure and one `test_domain_partition.py` failure that was a rounding-sensitive assertion and has since been rewritten (that file passes).


<a id="gallery"></a>

## 🖼️ Example gallery

All gallery examples of both packages, including the contact examples (`fea_03`, `fea_09`, `fea_10`, `rom_07`). Each script prints its own validation numbers and saves its figure.

<table>
<tr>
<td align="center"><a href="fea_engine/examples/gallery/fea_01_cantilever_beam.py"><img src="fea_engine/examples/gallery/fea_01_cantilever_beam.png" width="330"></a><br><sub><b>Cantilever</b>: von Mises stress and mesh convergence to the beam-theory value</sub></td>
<td align="center"><a href="fea_engine/examples/gallery/fea_02_hyperelastic_neo_hookean.py"><img src="fea_engine/examples/gallery/fea_02_hyperelastic_neo_hookean.png" width="330"></a><br><sub><b>Neo-Hookean</b>: large-strain hyperelasticity, load-deflection against a linear reference</sub></td>
<td align="center"><a href="fea_engine/examples/gallery/fea_03_hertzian_contact.py"><img src="fea_engine/examples/gallery/fea_03_hertzian_contact.png" width="330"></a><br><sub><b>Penalty contact</b>: block pressed onto a rigid circular obstacle</sub></td>
</tr>
<tr>
<td align="center"><a href="fea_engine/examples/gallery/fea_04_plasticity_strip.py"><img src="fea_engine/examples/gallery/fea_04_plasticity_strip.png" width="330"></a><br><sub><b>J2 plasticity</b>: plastic strain localises at the hole of a plate</sub></td>
<td align="center"><a href="fea_engine/examples/gallery/fea_05_modal_analysis.py"><img src="fea_engine/examples/gallery/fea_05_modal_analysis.png" width="330"></a><br><sub><b>Modal analysis</b>: first four mode shapes of a cantilever</sub></td>
<td align="center"><a href="fea_engine/examples/gallery/fea_06_topology_optimization.py"><img src="fea_engine/examples/gallery/fea_06_topology_optimization.png" width="330"></a><br><sub><b>Topology optimisation</b>: 86.8 % less compliance at fixed volume</sub></td>
</tr>
<tr>
<td align="center"><a href="fea_engine/examples/gallery/fea_07_pyvista_visualization.py"><img src="fea_engine/examples/gallery/fea_07_pyvista_2d_von_mises.png" width="330"></a><br><sub><b>PyVista contour</b>: von Mises stress on the deformed cantilever (optional <code>viz</code> extra)</sub></td>
<td align="center"><a href="fea_engine/examples/gallery/fea_08_pyvista_animation_sections_rom.py"><img src="fea_engine/examples/gallery/fea_08_rom_rank4.png" width="330"></a><br><sub><b>Full vs reduced</b>: POD rank 4, with the error field (optional <code>viz</code> extra)</sub></td>
<td align="center"><a href="fea_engine/examples/gallery/fea_09_sparse_obstacle_contact.py"><img src="fea_engine/examples/gallery/fea_09_sparse_obstacle_contact.png" width="330"></a><br><sub><b>Sparse exact contact</b>: <code>Quad4Laplace</code> membrane on an obstacle, contact set, route timing (<code>fea_09</code>)</sub></td>
</tr>
<tr>
<td align="center"><a href="fea_engine/examples/gallery/fea_10_bdf2_contact_dynamics.py"><img src="fea_engine/examples/gallery/fea_10_bdf2_contact_dynamics.png" width="330"></a><br><sub><b>BDF2 contact dynamics</b>: order 2 against order 1, bar impact (<code>fea_10</code>)</sub></td>
<td align="center"><a href="rom_engine/examples/gallery/rom_01_pod_basis.py"><img src="rom_engine/examples/gallery/rom_01_pod_basis.png" width="330"></a><br><sub><b>POD basis</b>: singular-value decay and reconstruction error</sub></td>
<td align="center"><a href="rom_engine/examples/gallery/rom_02_galerkin_parametric_rom.py"><img src="rom_engine/examples/gallery/rom_02_galerkin_parametric_rom.png" width="330"></a><br><sub><b>Parametric Galerkin ROM</b>: affine parametric reduction</sub></td>
</tr>
<tr>
<td align="center"><a href="rom_engine/examples/gallery/rom_03_greedy_frequency_training.py"><img src="rom_engine/examples/gallery/rom_03_greedy_frequency_training.png" width="330"></a><br><sub><b>Greedy training</b>: adaptive against uniform frequency sampling</sub></td>
<td align="center"><a href="rom_engine/examples/gallery/rom_04_balanced_truncation.py"><img src="rom_engine/examples/gallery/rom_04_balanced_truncation.png" width="330"></a><br><sub><b>Balanced truncation</b>: actual error stays below the certified bound</sub></td>
<td align="center"><a href="rom_engine/examples/gallery/rom_05_nonlinear_modal_rom.py"><img src="rom_engine/examples/gallery/rom_05_nonlinear_modal_rom.png" width="330"></a><br><sub><b>Nonlinear modal ROM</b>: 2 modes vs 51-DOF full-order beam, max error 3.3 %</sub></td>
</tr>
<tr>
<td align="center"><a href="rom_engine/examples/gallery/rom_06_dataset_diagnostics_uq.py"><img src="rom_engine/examples/gallery/rom_06_dataset_diagnostics_uq.png" width="330"></a><br><sub><b>Dataset diagnostics and UQ</b>: coverage and ensemble uncertainty</sub></td>
<td align="center"><a href="rom_engine/examples/gallery/rom_07_contact_rom_dynamics.py"><img src="rom_engine/examples/gallery/rom_07_contact_rom_dynamics.png" width="330"></a><br><sub><b>Contact ROM</b>: reduced dynamics against full order, error against basis size, greedy against random snapshots (<code>rom_07</code>)</sub></td>
</tr>
</table>

Click an image to open the script that produced it. `fea_08` also writes two animated GIFs; `fea_07` and `fea_08` need the optional `pyvista` package. Gallery figures are in `examples/gallery/` of each package.

<a id="batched"></a>

## ⚡ Important note: batched nonlinear element kernels and vectorised assembly

> [!IMPORTANT]
> **Wave 22, closed 2026-10-15. Opt-in, off by default.** The nonlinear elements `Quad4PlaneStressTL`, `Hex8SolidTL`, `Hex20SolidTL` and `Tet10SolidTL` can evaluate force and tangent for all elements of a block at once, and `FESystem` can assemble them without a Python loop over elements, into a sparse tangent that a sparse Newton solve uses. The loop versions remain the default and the reference; every batched result is tested against them. Measured on the author's machine (12 CPUs): kernel force 102x (Quad4), 47x (Hex8), 23x (Hex20) and 41x (Tet10) faster than the loop, agreeing to 8.4e-15 relative or better.

### Why it is needed

A nonlinear (total-Lagrangian) element has no precomputed matrix: its force and tangent depend on the current displacement, so they are recomputed in every Newton iteration. Per element and Gauss point the work is

```
F = I + (dN/dX)ᵀ u_e          deformation gradient
E = ½ (FᵀF − I)               Green-Lagrange strain
S = D : E                     second Piola-Kirchhoff stress (Saint Venant-Kirchhoff)
f_e = Σ_g w_g J0_g  B(u_e)ᵀ S           internal force
K_e = Σ_g w_g J0_g [ B(u_e)ᵀ D B(u_e) + geometric term(S) ]     tangent
```

The package's loop does this one element at a time in Python, and for small elements the interpreter, not the arithmetic, is the cost: the element is a few hundred flops wrapped in thousands of Python operations. The same is true of the scatter `K[np.ix_(g, g)] += K_e` into a dense `n × n` array, and of the dense `np.linalg.solve` that follows.

### Theory: batch over elements, fold over Gauss points

All elements of a block share the same shape-function derivatives per Gauss point in the reference configuration. The batched kernel therefore stacks the data as arrays with a leading element axis, coordinates `(E, n_nodes, dim)` and displacements `(E, ndof)`, and replaces the element loop by array contractions (einsum or batched matrix products) over axes `(E, G)`. It returns `(E, ndof)` forces and `(E, ndof, ndof)` tangents in the same local DOF order as the loop. Three ideas carry the speed-up and the scalability:

- **No interpreter per element.** The element loop becomes one NumPy call per kernel stage. This is the dominant gain for 2-D elements, where the arithmetic per element is tiny.
- **Folding the Gauss-point sum.** `K_e = Σ_g w_g Bᵍᵀ D Bᵍ` is evaluated as a single product with contraction length `G × 6` (the weights are folded into `D B`), instead of `G` small products. This was about 1.9x to 2.0x faster than the plain per-Gauss-point form on the 3-D hexahedron kernel, in the plan's measurement.
- **A cached geometry and a cached sparsity pattern.** The reference gradients `dN/dX` and `det(J0) w` come from `MeshTransformation` once per block. The CSR pattern of the assembled tangent and, for every entry of every element matrix, its position in the CSR data array are computed once (one `np.unique` over the flattened row/column keys). Each later tangent assembly is `bincount(positions, weights=K_e)`: no sorting, no COO-to-CSR conversion and no `n × n` array.

### Implementation

| Piece | Where | What it does |
|---|---|---|
| Batch protocol | `batched_kernels.py`; `Element.supports_batch` | An element opts in with `supports_batch = True` and `internal_force_batch` / `tangent_stiffness_batch`; a formulation without them is assembled by the loop, block by block, with the reason recorded in `FESystem.last_assembly_modes` |
| Chunking | `batched_kernels.chunk_slices` | A block is split so the estimated working memory stays under a budget (256 MB default) or under `chunk_size`; a Hex20 3x3x3 strain-displacement array alone is 78 kB per element, so an unchunked 10⁵-element block would need gigabytes. Chunked and unchunked results agree to summation order, not bit for bit |
| Elements | `elements/nonlinear_solids.py`, `elements/hex_tl.py` | Batched `Quad4PlaneStressTL` and `Tet10SolidTL`; new total-Lagrangian `Hex8SolidTL` and `Hex20SolidTL`, each written twice (a plain per-Gauss-point reference and the batched kernel) so the two can check each other |
| Assembly | `FESystem.assemble_internal_force` / `assemble_tangent_stiffness` | New keywords `vectorized`, `chunk_size`, `sparse`; `FESystem.batch_support_report()` says which blocks can be batched. Defaults reproduce the previous behaviour |
| Sparse Newton | `nonlinear_solver.py`, `NewtonOptions` | `sparse_tangent`, `lu_ordering` and `reuse_factorization` fields; sparse LU factorisation instead of a dense solve in the static, displacement-control, arc-length and transient drivers |
| Contact | `contact_batch.py`; `FESystem.add_contact_set` | `ContactSet2D` / `ContactSet3D`: penalty contact of many slave-master pairs in one call, with a KD-tree candidate search |
| Hyper-reduction | `FESystem.element_forces` / `element_tangents`; `rom_engine` `ECSW.fit_batched`, `ecsw_training_matrix_batched`, `nonlinearity_indicator_batched` | Element-level hooks, so ECSW training and the domain-partition indicator run without a Python loop over elements and snapshots |
| Constraints and torch | `FESystem.eliminate_constraints`; `batched_kernels_torch.py` | Elimination of homogeneous constraints; a torch version of the same kernels (CPU or CUDA) |

```python
fes.batch_support_report()                                   # which blocks can be batched, and why not
f = fes.assemble_internal_force(u, mat, vectorized=True)
K = fes.assemble_tangent_stiffness(u, mat, vectorized=True, sparse=True)   # scipy CSR
opts = NewtonOptions(sparse_tangent=True)                    # sparse Newton in the nonlinear drivers
```

### Measured advantages

Times from `paper/results/batched_kernels.json` (12 CPUs, NumPy 1.26.4, SciPy 1.15.3, Windows); the reference route is timed once, the batched route is the minimum of five runs; agreement is the maximum relative difference to the loop.

| Case | Elements / DOFs | Force | Tangent | Whole-system force | Sparse LU against dense solve | Agreement |
|---|---|---|---|---|---|---|
| `Quad4PlaneStressTL` | 1,600 / 3,362 | 102x | 64x | 112x | 12.7x | 8.4e-15 |
| `Hex8SolidTL` | 1,536 / 6,075 | 47x | 35x | 51x | 3.3x | 4.1e-15 |
| `Hex20SolidTL` | 432 / 6,951 | 23x | 17x | 22x | dense not run | 7.1e-15 |
| `Tet10SolidTL` | 768 / 4,131 | 41x | 53x | 28x | 1.8x | 4.2e-15 |

| Other batched paths | Measured |
|---|---|
| Contact search, KD-tree against loop | 89x (2-D, 200 pairs), 151x (3-D, 1,152 pairs); same pairs found |
| Contact force / tangent, set against per pair | 51x / 49x (2-D), 106x / 80x (3-D) |
| Hyper-reduction on 400 elements, 8 snapshots | training matrix 92x, reduced force 11x, nonlinearity indicator 76x; the same 13 elements selected by ECSW |
| Constraint elimination | 3% overhead on tangent and force assembly |
| Torch kernels, 1,536 Hex8 | CPU 3.3x (force) and 2.5x (tangent) over NumPy; CUDA 2.9x and 1.1x. The GTX 1050 was no faster than torch on the CPU for the one case timed |

### Limits and cautions

- **The gain depends on the element.** For a 2-D element the saving is mostly interpreter overhead, so the factor is large. A 3-D hexahedron is compute-bound: in the plan's measurement a batch-of-one call of the same kernel was only about 1.1x slower (tangent) than the batched one, so most of the 3-D gain over the loop comes from the folded kernel, the sparse tangent and the cheaper solve, not from removing the loop. At around 11,000 DOFs the sparse solve, not the assembly, dominates.
- **Single runs on one machine.** The figures show direction and order of magnitude, not precise ratios; the reference route was timed once.
- **Opt-in.** Nothing changes unless you ask. Path-dependent elements (those that need `state` or `iter_state`) are not batched.
- **Sparse ordering and factorisation reuse are not free wins.** The SuperLU symmetric-mode ordering did not help on a generic Hex20 mesh, and reusing the factorisation (modified Newton) can need more iterations; both are opt-in settings with full Newton as the default.
- **Not tested:** meshes of 10⁵ elements, batched friction, updating contact pairs during a solve. The batched `ContactSet` evaluates penalty contact only, with a smoothed penalty law available through `smoothing_length` (default 0 is the package law).
- `rom_engine` does not import `fea_engine`: its batched hooks take plain callables, `fn(elements, U_local)`.

Plan, research, per-item notes and every number above: [`fea_engine/docs/batched_nonlinear_kernels_plan.md`](fea_engine/docs/batched_nonlinear_kernels_plan.md); benchmark script: `paper/experiments/e10_batched_kernels.py`.

<a id="s-2"></a>

## ⚙️ Install

**Repository:** <https://github.com/abhi2k16/fea-rom-computation-suite>

### Clone the repository

Requires [Git](https://git-scm.com/downloads) and Python 3.9 or newer.

```bash
git clone https://github.com/abhi2k16/fea-rom-computation-suite.git
cd fea-rom-computation-suite
```

Using SSH instead (if you have a key set up with GitHub):

```bash
git clone git@github.com:abhi2k16/fea-rom-computation-suite.git
```

Create an isolated environment (recommended), then install both packages in editable mode so changes to the source take effect immediately:

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell refuses to run the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Anaconda: `conda create -n fea-rom python=3.11` then `conda activate fea-rom`.

To update an existing clone later: `git pull` from the repository folder.

### Install the packages

```
pip install -e fea_engine
pip install -e rom_engine
```

With the optional extras (plotting, tests, and the `fea` link for the ROM examples):

```
pip install -e "fea_engine[plot,dev]"
pip install -e "rom_engine[fea,dev]"
```

Python 3.9 or newer. Required: NumPy 1.22+ and SciPy 1.8+.

Check the install:

```
python -c "import fea_engine, rom_engine; print(fea_engine.__version__, rom_engine.__version__)"
```

**Optional dependencies**

| Need | Install |
|---|---|
| Plotting | `matplotlib` (the `plot` extra of `fea_engine`) |
| Mesh file I/O | `pip install meshio` |
| 3-D visualization (contours, deformed shape, animation, slices, ROM comparison, VTK export) | `pip install pyvista imageio` (the `viz` extra of `fea_engine`) |
| PyTorch, CPU only | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| PyTorch, NVIDIA GPU | `pip install torch --index-url https://download.pytorch.org/whl/cu126` (match the `cuXXX` to your driver) |

There is no generic `[torch]` extra on purpose, because the right wheel depends
on your hardware. A plain `import fea_engine` or `import rom_engine` never
needs torch, matplotlib or pyvista. See `USER_GUIDE.md` for known torch problems
(Pascal GPUs, the OpenMP error on Windows with Anaconda).

<a id="s-3"></a>

## 🚀 Quick start

A 0.4 x 0.2 m steel plate, clamped on the left and loaded on the right:

```python
import numpy as np
from fea_engine import Material, D_plane_stress, Quad4PlaneStress, FESystem
from fea_engine.mesh import rectangle_mesh

mat = Material(E=2.1e11, nu=0.3, rho=7850.0)
mesh = rectangle_mesh(Lx=0.4, Ly=0.2, nx=24, ny=12)
sys = FESystem(mesh, Quad4PlaneStress(), thickness=0.02)
sys.assemble_stiffness(D_plane_stress(mat), thickness=0.02)

tip = mesh.nodes_on_line(axis=0, value=0.4)
sys.add_nodal_force(tip, dof_index=1, total_force=-20000.0)
sys.fix_dofs(mesh.nodes_on_line(axis=0, value=0.0), [0, 1])

U = sys.solve_static()          # mean tip deflection about -1.797e-4 m
```

Every `fea_engine` problem has the same shape: **material, mesh, element,
`FESystem`, then assemble, apply loads and constraints, solve.** Swap the
element or the `solve_*` call and the rest stays the same.

Then reduce it:

```python
from rom_engine import PodBasis, GalerkinROM

sys.assemble_mass(mat.rho * np.eye(2))
# snaps: columns are full-order solutions at several load levels
basis = PodBasis().fit(snaps, n_modes=6, M=sys.M)
rom = GalerkinROM(basis).reduce_system(sys.K, M=sys.M, F=sys.F)
x_full, q = rom.solve_static()  # 650 unknowns reduced to 6
```

`API_REFERENCE.md` has the complete runnable versions (including how to build
`snaps`), natural frequencies, a nonlinear beam, affine sweeps, frequency
response and balanced truncation, each with the output it printed.

<a id="s-4"></a>

## 🔄 A typical workflow

1. **Full-order model (`fea_engine`):** mesh, material, element, loads, constraints, solve. Check one result against a hand calculation or analytic solution.
2. **Snapshots:** solve at a handful of load levels or parameter values (`solve_batched` reuses one factorisation for many loads).
3. **Reduce (`rom_engine`):** POD basis, then Galerkin projection. For parameter sweeps add `AffineDecomposition`; for dynamics try modal or frequency-domain ROMs.
4. **Verify:** compare the reduced answer with a full-order solve at points not used for training. The packages report energy captured, residuals and, for some methods, certified error bounds.
5. **Use online:** run the small reduced model thousands of times for design sweeps, optimisation or uncertainty studies.

<a id="s-5"></a>

## 🖥️ CPU, multi-core and GPU

```python
sys = FESystem(mesh, elem, thickness=0.02, backend="torch", device="cuda")   # GPU solve
```

SciPy is the default and torch/CUDA are opt-in. Neither package schedules its
own threads; multi-core use comes from the BLAS library NumPy links against.

- **CPU threads:** set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` before NumPy is imported.
- **`run.py`:** runs any script with a chosen core count without editing it (see the docstring at the top of `run.py`).
- **GPU:** only pays off for large sparse systems; small models are usually faster on the CPU. `backend="auto"` chooses for you above 5000 free DOF.
- **Sweeps:** the frequency/amplitude sweeps in `rom_engine.attractor` accept `n_jobs=` for multi-process execution.

Details, measured examples and a "when does a GPU help" table are in the
"Multi-core, CPU and GPU (PyTorch) usage" section of `USER_GUIDE.md`.

<a id="s-6"></a>

## 🧪 Examples

| Where | What |
|---|---|
| `fea_engine/examples/main.py` | Eight problems run end to end against closed-form references. |
| `fea_engine/examples/gallery/` | Cantilever beam, hyperelasticity, Hertzian contact, plasticity, modal analysis, topology optimisation (with a README). |
| `fea_engine/examples/` | Buckling, arc-length snap-through, 3-D frames, shells, higher-order elements, sparse vs dense timing, SciPy vs torch benchmark. |
| `rom_engine/examples/gallery/` | POD basis, parametric Galerkin ROM, greedy frequency training, balanced truncation, nonlinear modal ROM, dataset diagnostics (with a README). |
| `rom_engine/examples/` | Two-region beam affine sweep, frequency sweep, certified bounds, mode-shape recovery, plate modal identification. |

Run an example from its package folder, for instance
`cd fea_engine && python examples/main.py`.

<a id="s-7"></a>

## 📚 Documentation

| File | Contents |
|---|---|
| [USER_GUIDE.md](USER_GUIDE.md) | Getting started, concepts, a tour of every capability with worked examples, multi-core and GPU usage, performance numbers. |
| [API_REFERENCE.md](API_REFERENCE.md) | Every public class, function and method with signature and purpose, a task-to-API map, and eight executed examples. |
| `fea_engine/README.md`, `rom_engine/README.md` | Full module references and validation records. |
| `fea_engine/docs/` | Roadmaps and implementation notes (nonlinear FEM, shells, mesh grading, meshing alternatives, TensorMesh analysis). |
| `rom_engine/docs/` | Roadmaps and design notes (classical MOR, frequency-domain ROM, error bounds and greedy training, Loewner identification, nonlinear surrogate ROMs, differentiable correction, testing methodology). |
| `fea_engine_Technical_Notes.*`, `rom_engine_Technical_Notes.*` | Technical notes (Word and PDF). |
| `fea_engine_rom_engine_coupling.html` | How the two packages couple and the torch-optional pattern. |
| `hpc_translation_roadmap.html` | Plan for porting the numerical core to HPC. |
| `module_review_*.md` | Review notes. |

<a id="s-8"></a>

## 🗂️ Repository layout

```text
computation-suite/
  fea_engine/            finite element package   (src/, tests/, examples/, docs/)
  rom_engine/            reduced-order modeling   (src/, tests/, examples/, docs/)
  run.py                 run any script with a chosen number of CPU cores
  USER_GUIDE.md
  API_REFERENCE.md
  module_review_*.md
  *_Technical_Notes.*
```

Both packages use the standard `src/` layout and are pip-installable.

<a id="s-9"></a>

## ✅ Tests

```
cd fea_engine  && pytest tests/ -q
cd rom_engine  && pytest tests/ -q
```

Tests that need PyTorch skip themselves when it is not installed. The test
suites compare against analytic solutions, independent implementations and
theorems (for example passivity and certified bounds); `rom_engine/docs/testing_methodology.md`
describes the approach.

<a id="s-10"></a>

## ⚠️ Notes and limits

- **Units:** the packages are unit-agnostic. Use one consistent system, such as SI.
- **Reduced results:** always compare a reduced result with at least one full-order solve before relying on it.
- **Torch coverage is selective:** `backend="torch"` currently affects `solve_static()` and a few named routines; modal, buckling and the explicit transient drivers stay SciPy/NumPy-only.
- **Research-grade modules** in `rom_engine` (neural operator, latent ODE, reduced-basis operator, ensemble UQ, differentiable correction) are prototypes.
- **Gmsh:** Gmsh-based geometry support has been removed; use the structured mesh generators, `meshio`, or mesh files from external tools.

<a id="s-11"></a>

## 📄 License

Both packages declare the MIT license in their `pyproject.toml`. There is no
`LICENSE` file in this folder yet; add one before publishing.
