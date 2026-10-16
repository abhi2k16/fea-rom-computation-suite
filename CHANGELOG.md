# Changelog

All notable changes to `fea_engine` and `rom_engine`. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) (`rom_engine` is pre-1.0, so minor releases may change behaviour).

## Unreleased

### Added (rom_engine, cached contact-condition indicator)
- `ContactConditionIndicator(K, B, f, weights=(1, 0, 0), max_dense_bytes=1e9)`: the indicator of `contact_condition_indicator` (kept unchanged as the reference) with the force estimate `(B^T)^+ (K u - f)` prepared once and only when `a2` or `a3` is nonzero, by the cheapest exact route: `identity` and `selection` need no solve, `normal_equations` is one sparse LU of `B B^T` for a sparse full-row-rank `B` (checked on random vectors), `dense_pinv` caches the dense pseudo-inverse otherwise (refused above `max_dense_bytes`). Accepts trajectories `(n, T)` (sum over the columns, per-step `f` and `c` allowed); `for_parameter(c_of_p)` gives the `indicator(p, u)` for `greedy_contact_sampling`. Tests: `rom_engine/tests/test_contact_indicator.py` (37); timing: `rom_engine/tests/contact_indicator_timing.py`.

### Added (rom_engine, greedy contact sampling for trajectories)
- `greedy_contact_sampling`: `fom_solve` may return trajectories `(n, T)` and `(m, T)` (columns concatenated in selection order); new keywords `rel_tol` (stop when the largest indicator is at most `rel_tol` times the largest after the first round: Algorithm 2 line 15 of Balajewicz et al., read as `<=` because the printed `>` is most likely a typo) and `pass_round` (calls `make_rom(U, L, n_iter)`); `GreedyContactHistory` gained `n_columns` and `stop_reason`. New helpers `trajectory_indicator` (sums a single-state indicator over the time steps) and `linear_basis_schedule(p_first, p_last, n_last)` (the report's `p = floor(2 + (n - 1) 98/9)` is `linear_basis_schedule(2, 100, 10)`). Vector snapshots, `tol` and the two-argument `make_rom` behave as before. Tests added to `rom_engine/tests/test_contact_greedy.py` (10 new, 13 in all).

### Added (rom_engine, dynamic contact ROM)
- `ContactROM.solve_dynamic(force, c, dt, n_steps, u0=None, v0=None, scheme="bdf2", warm_start=True, lift=True)`: reduced BDF2 contact dynamics on `M_r = U^T M U`, `A_r`, `B_r` (new optional `M=` argument of `ContactROM`; the static interface is unchanged). The two effective reduced matrices and their LCP matrices are factored once; the previous step's active set warm-starts the next LCP (`lcp_active_set` and `solve_lcp` gained an optional `active0`). Every step's reduced contact conditions are verified (relative tolerance 1e-4); a violation, or a failing LCP, stops the loop and is reported (`ok`, `failed_step`, `message`, rows from that step NaN). Returns `ContactROMDynamicResult` (reduced and lifted displacements, multipliers, gaps, velocities, iteration counts, `kkt_residual`). Tests: `rom_engine/tests/test_contact_dynamic_rom.py` (13).

### Added (fea_engine, BDF2 contact dynamics)
- `solve_contact_dynamics(..., scheme="bdf2")`: second-order BDF2 time stepping (BDF1 first step; general sparse or dense `M`, `K`, `B`; both effective matrices factored once; contact enforced at every step). `scheme="two_step"` (the default) is unchanged. New keywords `scheme`, `return_info` (adds a `ContactDynamicsInfo` with the velocities, the per-step iteration counts and, if a contact solve failed, the step, reason and suggestion; rows from the failed step on are NaN), `route`, `fallback`; `method="pdas"` (BDF2 only) solves each step with `solve_contact_static_sparse` and warm-starts from the previous active set. Not contact-stabilised: an impact test shows BDF2 gaining up to 0.7% above the initial energy during the impact. Newmark is not provided. Tests: `tests/test_contact_dynamics_bdf2.py` (14).

### Added (fea_engine, scalar Laplace / Poisson element)
- `Quad4Laplace` in new `elements/scalar.py` (registry key `quad4_laplace`, exported from `fea_engine` and `fea_engine.elements`): 4-node bilinear scalar-field element, one DOF per node (named `"u"`), for `-div(k grad u) = q` (membrane, steady heat conduction) and, with the mass matrix, the wave and heat equations. Generic Gauss loop; `stiffness` takes a scalar `k` or a 2 x 2 matrix, `mass` / `lumped_mass` (HRZ) a scalar or 1 x 1 density, `source_vector` the consistent load of a source (scalar or nodal values). Not a plane-stress subclass. Tests: `tests/test_quad4_laplace.py` (17). Triangles, hexahedra, variable coefficients and flux recovery are not provided.

### Added (fea_engine, sparse unilateral contact solver)
- `solve_contact_static_sparse(K, f, B, c, route="auto", active0=None, fallback=False, max_iter=100, tol=1e-10, max_dense_bytes=1e9)` in `contact_solvers.py`: primal-dual active set (semismooth Newton) for sparse `K` without forming `B K^-1 B^T`. Optional routes: `"reduced"` (B selects nodes, e.g. `B = I`), `"saddle"` (general sparse B, saddle system with the active rows), `"condensed"` (the existing dense routes); `"auto"` picks `reduced` or `saddle` and never `condensed`. A failing route (cycling, iteration limit, singular saddle system, route not applicable) returns `converged=False` with `reason`, `message` and `suggestion` (the next route, with the dense route's memory estimate); `fallback=True` follows the suggestions (the dense route only if it fits `max_dense_bytes`). Warm start by `active0` (the previous result's `active_mask`). `ContactResult` gained optional fields `route`, `reason`, `message`, `suggestion`, `active_mask`, `fallbacks` and the alias `ok`; `solve_contact_static(..., method="pdas")` calls the new solver. Existing functions and defaults unchanged. No convergence guarantee for stiffness matrices that are not M-matrices (the failure flag exists for that). Tests: `tests/test_contact_solvers_sparse.py` (28); timing: `tests/sparse_contact_timing.py`.

### Added (documentation and examples, contact module)
- A "Contact module" section in `README.md`, `fea_engine/README.md` and `rom_engine/README.md` (modules, elements and methods added or changed, measured claims, gallery figures, tests), replacing the earlier callouts.
- `USER_GUIDE.md`: two new chapters after "Nonlinear analysis", "Unilateral contact: exact solvers, dynamics and contact ROMs" and "Batched nonlinear element kernels and vectorised assembly" (code in both was run in the sandbox), plus rows in the ROM chooser, the quick decision guide and the performance table, and the gallery description.
- Gallery examples `fea_engine/examples/gallery/fea_09_sparse_obstacle_contact.py`, `fea_10_bdf2_contact_dynamics.py` and `rom_engine/examples/gallery/rom_07_contact_rom_dynamics.py`, each with its figure; the gallery READMEs list them. Figures were produced in the sandbox (2 CPUs); not yet run on the author's machine.

### Wave 23 closed (2026-10-16)
Contact modelling and contact ROMs are complete: items 211-216 as listed above and in `fea_engine/docs/consolidated_future_roadmap.md`; item 217 closed as a decision (keep `Quad4MindlinPlate`, no new element). Full suites on the author's machine: `fea_engine` 1553 passed, 12 skipped, 2 xfailed; `rom_engine` 658 passed, 1 skipped, 2 failed (the known operator-inference accuracy failure, and a rounding-sensitive count in `test_domain_partition.py`, whose test now checks where the weak elements can be; that file passes, the full `rom_engine` suite was not re-run). Not done: friction, Newmark, parametric dynamic prediction, a dense-route run at 39,601 DOFs.

### Wave 22 closed (2026-10-15)
Batched nonlinear element kernels (opt-in) are complete: items 198-210 as listed below and in `fea_engine/docs/batched_nonlinear_kernels_plan.md`. Loop versions remain the default. Open at closing: the full pytest run on the author's machine had not been reported; not tested: 10^5-element meshes, batched friction, pair updating during a solve.

### Added (paper experiments, batched kernels)
- `paper/experiments/e10_batched_kernels.py` (+ `run_all.py` entry, README row): one block per element family, 2-D and 3-D contact, hyper-reduction, constraint-elimination overhead and the torch kernels, with the maximum difference reported next to every ratio; writes `paper/results/batched_kernels.json` (environment included). A run on the 2-core sandbox is stored there; rerun on your machine to replace it (the torch block runs only where torch is importable).

### Added (fea_engine, torch backend of the batched kernels)
- `batched_kernels_torch.py`: `tl_internal_force` / `tl_tangent` (total-Lagrangian kernels written once for numpy and torch, 2-D and 3-D), `internal_force_batch_torch` / `tangent_stiffness_batch_torch` (numpy in / numpy out, `device=`), `assemble_internal_force_torch` / `assemble_tangent_stiffness_torch` (structural blocks of Quad4PlaneStressTL, Hex8SolidTL, Hex20SolidTL, Tet10SolidTL; contact, springs, constraints and element state refused). `tests/test_batched_kernels_torch.py` (30 tests): 29 passed, 1 skipped (the torch-absent case), 0 failed on the user's machine, including the CUDA test. No timing.

### Added (fea_engine, linear constraints in nonlinear analysis)
- `constraint_elimination.EliminatedSystem` and `FESystem.eliminate_constraints()`: the linear constraints of `add_constraint` / `tie` are eliminated exactly (`u = T q`, `f_q = T^T f(Tq)`, `K_q = T^T K T`, dense or CSR). `solve_nonlinear_static` and `solve_nonlinear_arc_length` now accept constrained systems automatically (displacements returned in the full numbering); `solve_nonlinear_displacement_control` works on the explicit object. Homogeneous constraints and zero prescribed values only; not the transient driver. `tests/test_constraint_elimination.py` (9 tests); constrained patch test (two halves tied vs one piece, large displacement) agrees to 9e-16.

### Added (fea_engine, batched Tet10SolidTL)
- `Tet10SolidTL.internal_force_batch` / `tangent_stiffness_batch` (the generic total-Lagrangian array kernels with the element's own 4-point rule) and the geometry-cache hook `own_batch_geometry` (the cache takes the element's own `_reference_geometry_batch` instead of `MeshTransformation`'s 8-point rule). `tests/test_tet10_tl_batched.py` (14 tests); timing script `tests/batch_timing_tet10.py` (768 elements: kernel force 20.6x, tangent 33x, system force 30x, CSR tangent 10x faster than the dense tangent assembly, 2 cores).

### Added (fea_engine, batched 3-D contact)
- `contact_batch.ContactSet3D` (node-to-triangle penalty pairs in array form; the law, force split and consistent Gauss-Newton tangent of `NodeToSurfaceContact3D`, or its central-difference tangent on request; optional named smoothing and nearest-candidate selection, both off by default), `candidate_pairs_3d` (KD-tree broad phase) and `closest_point_on_triangles` (array form of the scalar region test, same precedence). `tests/test_contact_set_3d.py` (21 tests); timing script `tests/batch_timing_contact_3d.py` (1152 pairs: contact force 92x, tangent 69x, system CSR tangent 6.4x faster than one element per pair, 2 cores). Note: with the consistent tangent Newton stalls on the test model for k_p >= 700 (also with the per-element registration), through active-set chattering of the plain penalty ramp; `consistent_tangent=False` or `smoothing_length` of 0.005 or more converged in every case tried.

### Added (fea_engine, batched 2-D contact)
- `contact_batch.ContactSet2D` (many node-to-segment penalty pairs in one array evaluation; the law, force split and central-difference tangent of `NodeToSegmentContact2D`; optional named smoothing, off by default), `candidate_pairs_2d` (KD-tree broad phase, same pairs as `find_contact_pairs_2d`) and `FESystem.add_contact_set` (force, dense and CSR tangent). `tests/test_contact_set_2d.py` (19 tests); timing script `tests/batch_timing_contact.py` (80 pairs: contact force 20x and tangent 21x faster than one element per pair, system CSR tangent 4.8x, 2 cores).

### Added (fea_engine and rom_engine, hyper-reduction hooks)
- `FESystem.element_forces(u, mat, elements=None, vectorized=None, chunk_size=None)`, `element_tangents(...)`, the callback forms
  `element_forces_local(elements, U_local, mat)` / `element_tangents_local(...)`, `element_dofs(elements=None)` and `n_elements`:
  per-element internal forces (E, ndof) and tangents (E, ndof, ndof) of a chosen subset, using the batched kernels where a block
  supports them and the per-element loop otherwise. Rows scatter to `assemble_internal_force` / `assemble_tangent_stiffness`
  (no contact or springs). `tests/test_element_response_hooks.py` (9 tests).
- `rom_engine`: `ecsw_training_matrix_batched`, `ECSW.fit_batched` and keyword-only `elem_forces_fn` / `elem_tangents_fn` on `ECSW`,
  `nonlinearity_indicator_batched`: one callback call per snapshot (or per reduced force) instead of one per element. The per-element
  functions are unchanged. `rom_engine/tests/test_hyper_reduction_batched.py` (8 tests); timing script
  `rom_engine/tests/hyper_reduction_timing.py` (measured 92-112x for the training matrix, 74-87x for the indicator on a 400-element
  Quad4PlaneStressTL plate, 2 cores).

### Added (fea_engine, elements)
- `Quad4PlaneStressTL` (element `"quad4_plane_stress_tl"`, `elements/nonlinear_solids.py`): 4-node plane-stress quadrilateral with
  Total-Lagrangian geometric nonlinearity and a St. Venant-Kirchhoff law (`material.D_plane_stress` on the Green-Lagrange strain),
  analytic tangent `B_nl^T D B_nl + G^T S G` (Nikiema et al. 2026, Eq. 18 in 2-D). Full 2x2 integration, no hourglass control.
  `tests/test_quad4_tl.py` (7 tests): tangent against central differences at small and large displacement, symmetry, linear limit,
  zero force under rigid rotation, exact uniform-stretch force, equality with `TrussTL2D` for a nu = 0 strip at 40% stretch, and a
  cantilever through `solve_nonlinear_static`. Known limit: like `Quad4PlaneStress`, it shear-locks in bending on slender meshes
  (80x4 mesh of the BE7 cantilever: linear Q4 tip 3.8 against 33.2 analytic), so beam-theory benchmarks are not valid for it.
  `solve_nonlinear_static` ignores `FESystem(thickness=...)`; pass `thickness=` to the solver call.

### Added (fea_engine, 3-D total-Lagrangian hexahedra)
- `Hex8SolidTL` and `Hex20SolidTL` (`elements/hex_tl.py`; registered as `"hex8_solid_tl"`, `"hex20_solid_tl"`): 8- and 20-node
  hexahedra with Total-Lagrangian geometric nonlinearity and a St. Venant-Kirchhoff law (`mat` = 6x6 `D_solid3d` matrix or the
  `Material`), a per-Gauss-point reference implementation and independently written batched kernels (`internal_force_batch`,
  `tangent_stiffness_batch`) that work with `FESystem(vectorized=True)`, the CSR tangent and `sparse_tangent=True`. `gauss_order`
  constructor argument (defaults 2 and 3). `Hex20SolidTL(gauss_order<3)` warns: the 2x2x2 rule has 6 spurious zero-energy modes
  per isolated element and a single-row mesh keeps them (singular tangent at zero, Newton raises at load step 1; meshes with two
  elements across the width or the thickness had none).
  Measured in the package (2 CPU cores, best of 3): batched against the per-Gauss-point reference loop, kernel force / tangent
  41.6x / 29.3x for 1536 Hex8 elements, 19.3x / 13.3x for 720 Hex20 elements (11,319 DOFs); `FESystem` force 50.0x and 20.2x;
  agreement 2e-16 to 1.7e-14 relative. Locking, 60x12x2 mm strip against Timoshenko: Hex8 0.282, 0.575, 0.804 (meshes 12x2x2,
  24x2x2, 48x2x4), Hex20 2x2x2 0.928 to 0.979, Hex20 3x3x3 0.897 to 0.974. Large-deflection cantilever (P L^2/(E I) = 1, nu = 0)
  against the elastica 0.3017 / 0.0561: Hex20 0.3016 / 0.0564 (3x3x3, 12x1x1) and 0.3020 / 0.0565 (2x2x2, 12x1x2); Hex8 locks
  (0.080 / 0.004). The values agree with the sibling project's to three decimals; the project's "0.1641" for single-row 2x2x2
  Hex20 was an unconverged state (its script prints 0.7518 now), which the package reports as a Newton failure.
  Tests: `test_hex_tl.py` (25), `test_hex_tl_batched.py` (27), `test_hex_tl_validation.py` (26); `tests/batch_timing_hex.py`.
  Use the Newton tolerance 1e-7 for stiff 3-D cantilevers (residual floor about 2e-9). Not done: neo-Hookean or other laws,
  batched path-dependent state.

### Added (fea_engine, sparse tangent and sparse Newton path)
- `FESystem.assemble_tangent_stiffness(..., sparse=True)` returns a `scipy.sparse.csr_matrix` (default `sparse=False` is the
  unchanged dense array). Batched blocks scatter through a per-block CSR pattern and entry-to-data position map built once
  (`batched_kernels.SparsePatternCache`, `build_sparse_pattern`, `scatter_add_csr_data`); loop blocks, contact elements and
  springs are added as COO triplets. The pattern survives mesh motion and is rebuilt if the element-to-DOF map changes.
- `sparse_tangent=False` and `lu_ordering=None` keywords on `solve_nonlinear_static`, `solve_nonlinear_displacement_control`,
  `solve_nonlinear_arc_length` and `solve_nonlinear_transient` (CSR free block, SuperLU; arc length factors once per iteration
  for its two right-hand sides; not available with `backend="torch"`); `reuse_factorization=False` on `solve_nonlinear_static`
  (modified Newton: tangent factored once per attempt, dropped when an iteration fails to reduce the residual). Matching
  `NewtonOptions` fields `sparse_tangent`, `lu_ordering` (NATURAL, MMD_ATA, MMD_AT_PLUS_A, COLAMD, symmetric) and
  `reuse_factorization`. Defaults reproduce the previous dense calls exactly.
  Measured (2 CPU cores, Quad4PlaneStressTL plates, best of 2-3): 3362 DOFs, dense solve 609 ms against sparse LU 17.9 ms
  (34x); 10,082 DOFs, 14.0 s against 124 ms (113x); CSR tangent assembly 15.4 ms against 31.7 ms for the batched dense one
  at 3362 DOFs. End-to-end `solve_nonlinear_static` (3 load steps, mildly nonlinear, tol 1e-8) on a 20x20 plate:
  2.86 s loop, 0.25 s batched dense, 0.06 s batched sparse; 40x40 plate: 5.23 s batched dense, 0.25 s batched sparse;
  results agree to 1e-14 relative. A 100x100 plate (20,402 DOFs; the dense tangent alone would need 3.3 GB) assembles and
  solves in 0.18 s + 0.26 s with a 51 MB peak (tracemalloc). `tests/test_sparse_newton.py` (22 tests). The symmetric-mode LU
  ordering is a measured opt-in: it gave no gain on a generic Hex20 box mesh in the planning study.

### Added (fea_engine, batched Quad4PlaneStressTL and vectorized assembly)
- `Quad4PlaneStressTL.internal_force_batch` / `tangent_stiffness_batch` (any Gauss order, thickness; Gauss-point sum folded into
  one matrix product) and `FESystem.assemble_internal_force` / `assemble_tangent_stiffness` with new keyword arguments
  `vectorized=None` and `chunk_size=None`, plus the attribute `FESystem.batched` (default False) and
  `FESystem.last_assembly_modes` ("batched" or "loop (reason)" per block). Blocks without batch kernels, with path-dependent
  state, contact elements and springs stay on the per-element loop. `batched_kernels.scatter_add_matrix`. Default behaviour is
  unchanged: with no new argument the loop runs and the result type is the same dense array. Batched and loop results agree to
  about 1e-15 relative (tests require 1e-12). Measured on 2 CPU cores, 324 elements / 722 DOFs, best of 5: internal force 118.9 ms
  (loop) against 1.09 ms (batched), 109x; dense tangent 150.0 against 7.87 ms, 19x (includes the dense scatter); 1600 elements /
  3362 DOFs, best of 3: 135x and 24x. One machine, so orders of magnitude, not exact ratios.
  `tests/test_quad4_tl_batched.py` (26 tests), including the golden loop Newton path reproduced to 1e-9.

### Added (fea_engine, batched-kernel foundation; no behaviour change)
- `batched_kernels.py`: the opt-in batch-method protocol (`supports_batch`, `internal_force_batch`, `tangent_stiffness_batch`),
  memory-bounded chunking (`chunk_slices`, `map_chunks`, byte estimates), a vectorised element-DOF map and bincount scatter, and
  `BlockGeometryCache` (reference shape gradients and JxW from `MeshTransformation`, rebuilt when nodes, connectivity or Gauss
  order change). `FESystem.block_geometry`, `clear_block_geometry` and `batch_support_report`; `Element.supports_batch = False`.
  No element implements the batch methods yet and no assembly path calls them, so existing results are unchanged.
  Tests: `tests/test_batched_kernels_foundation.py` (18), `tests/test_golden_loop_baselines.py` (10, frozen loop results for
  `Quad4PlaneStressTL` and `Tet10SolidTL`), helpers `tests/batch_conformance.py` and `tests/batch_timing.py`.
  Plan: `fea_engine/docs/batched_nonlinear_kernels_plan.md`.

### Added (fea_engine, visualization)
- `visualization/pyvista_plot.py` (optional `pip install fea_engine[viz]`, PyVista): `to_pyvista` (Mesh / MultiBlockMesh to
  `pyvista.UnstructuredGrid`, nine cell types), `displacement_vectors`, `add_displacement`, `deformed_grid`, `plot_field`
  (nodal/element field contours, optionally on the deformed mesh), `plot_deformed`, `save_vtk`, `set_off_screen`.
  `tests/test_pyvista_plot.py` (31 tests; 28 run without a display) checks the VTK node order of the quadratic cells.
  Example `examples/gallery/fea_07_pyvista_visualization.py`.
- `visualization/pyvista_animation.py` (`animate_history`, `animate_mode`, `mode_history`: GIF/MP4, colour limits and scale fixed
  over all frames), `visualization/pyvista_sections.py` (`slice_field`, `clip_field`, `threshold_field`, `plot_slice`,
  `plot_clipped`, `plot_threshold`), `visualization/pyvista_comparison.py` (`compare_fields`, `field_error`: full-order, reduced-order
  and error panels with relative L2 / max metrics). `plot_field` and the other plot functions accept a callable `field(U)`.
  Tests `test_pyvista_animation.py`, `test_pyvista_sections.py`, `test_pyvista_comparison.py` (64 PyVista tests in all; 53 run without a
  display); the display fixture lives in `tests/conftest.py`. Example `examples/gallery/fea_08_pyvista_animation_sections_rom.py`.

### Added (fea_engine, contact)
- `contact_solvers.py`: exact multi-node Signorini contact on plain arrays. LCP solvers (`lcp_active_set`, `lcp_lemke`,
  `lcp_projected_gauss_seidel`, `solve_lcp`), `solve_contact_static`, `solve_contact_dynamics` (implicit two-step scheme)
  and `solve_contact_newton` (sequential linearisation). Frictionless, constant gap map. (`solve_contact_static_sparse`, the sparse
  route, is listed under Unreleased.)
- `mortar.py`: `dual_mortar_operator` and `MortarTie2D` (tied and unilateral dual-mortar coupling of non-matching meshes,
  straight 2-D interfaces, linear segments, small displacements).
- `NodeToSurfaceContact3D` (element `"node_to_surface_contact_3d"`), `closest_point_on_triangle_3d`, `find_contact_pairs_3d`.

### Added (rom_engine, contact)
- `complementarity.py` (dense LCP copy, `solve_reduced_contact`), `contact_dual_basis.py` (NMF dual basis, `ContactROM`,
  rank/inf-sup report), `contact_greedy.py` (contact-condition greedy indicator), `contact_hyper_reduction.py`
  (contact-aware ECSW, constraint-element completion, NNLS force reconstruction, basis enrichment, reduced contact Newton),
  `domain_partition.py` (static condensation of the linear zone plus an element nonlinearity indicator designed here,
  since the source paper withholds its own), `contact_validation.py` (max-normalised error metrics).
- `friction_contact_models.py`: `JenkinsContact` (macro-slip stick/slip/lift-off elements for harmonic balance, exact
  Jacobian). `forced_response_hbm.py`: `ForcedHBM`, pseudo-arclength `forced_response_continuation`, `equivalent_resonance`,
  `equivalent_properties_vs_force`.
- `interface_reduction.py`: dual Craig-Bampton and its assembly, joint interface modes, Gram-Schmidt interface modes,
  Fourier and multi-stage Fourier interface bases, trial vector derivatives, frequency/MAC metrics.
- `component_pod.py` (component POD with mortar coupling), `operator_inference.py` (Craig-Bampton-structured SPD
  operator inference with LCP contact dynamics), `hyper_reduced_hbm.py` (harmonic ECSW and the residual error
  indicator of Morsy, Kast and Tiso 2022), `rubin_dual_basis.py` (Rubin plus dual modes, `reidentify_polynomial_force`),
  `contact_comparison.py` (comparison harness) and `examples/contact_rom_comparison.py`.
- `torch_contact_bases.py` (opt-in, not exported). Its 5 tests (`tests/test_torch_contact_bases.py`) pass against the NumPy
  originals on the maintainer's PyTorch machine (2026-10-13; 16 of 16 together with `test_torch_system_bases.py`).

### Added (rom_engine)
- System-driven bases for geometrically nonlinear structures (`modal_derivatives.py`): `static_modal_derivatives`
  and `dynamic_modal_derivatives` from a tangent-stiffness callable (central differences, one LU factorisation),
  `modal_interaction_weights` / `select_modal_derivatives` (maximum modal interaction) and
  `augment_with_derivatives` (M-orthonormal augmented basis).
- `dual_modes`: membrane enrichment from nonlinear static solves of the full model. The ranking rule is
  reconstructed from a partly garbled source equation; the docstring says so.
- `enhanced_enforced_displacement_rom` / `EnhancedEnforcedDisplacementROM` (`nonlinear_rom.py`): polynomial reduced
  force identified from reduced tangent probes only; pair probes (O(m^2) evaluations) already identify every term.
- `quadratic_manifold.py`: `QuadraticManifold` (`u = Phi q + 1/2 Omega(q,q)`, configuration-dependent reduced
  mass, convective force), `fit_quadratic_manifold`, `manifold_from_derivatives`, `lifted_training_states`,
  `ManifoldECSW` (hyper-reduction trained without full-order simulation) and `QuadraticManifoldROM`.
- `SlowFastROM` (`slow_fast.py`), `nnm_backbone_in_subspace` (`subspace_nnm.py`), `mode_sensitivities` /
  `updated_modes` (`parametric_modes.py`).
- `cms.py`: `free_interface_cms` (Rubin, MacNeal; requires a non-singular K) and `reduce_interface`
  (characteristic constraint modes).
- `validation.py`: `compare_bases`, `effective_speedup`, `break_even_queries`.
- `torch_system_bases.py`: opt-in PyTorch counterparts (`torch_*`) of the above, side by side with the NumPy
  versions; not exported from the package. Tests: `tests/test_torch_system_bases.py` (skip without torch).
- Measured on a 20-element clamped-clamped geometrically nonlinear beam loaded along mode 1 to three thicknesses
  of deflection (relative error against the full nonlinear solve): 3 modes alone 6.8%; with modal derivatives
  (9 vectors) 0.57%; dual modes 1.3%; quadratic manifold (still 3 coordinates) 0.57%. Tests: `test_modal_derivatives`, `test_dual_modes_eed`, `test_quadratic_manifold`,
  `test_slow_fast`, `test_subspace_nnm`, `test_free_interface_cms`, `test_parametric_compare`.

- `enhanced_enforced_displacement_rom(..., mode_scales=)`: per-coordinate probe sizes (array or callable of the
  amplitude). Needed for modal-derivative coordinates, which grow like the square of a mode amplitude; probing them
  at the mode amplitude gave a 91% static error in our beam test.
- `examples/system_driven_bases_timing.py` now also runs transient releases and a clamped corotational plate
  (`tests/shell_plate_fixtures.py`), resumable with `--cache`. Measured: manifold + ECSW 9.0x on a 100-element beam
  transient and 4.0x / 10.0x on 4x4 / 6x6 plate transients; Galerkin 1.0x to 1.6x; polynomial force fastest per run
  but 124 s to 274 s offline and failed on the beam transient. 6x6 plate accuracy (7.5% for every model) is unexplained.
- `examples/system_driven_bases_timing.py` and measured timings (2 CPU cores, best of 3) against a full-order
  load-stepped Newton solve over a sweep of 8 loads: quadratic manifold with ECSW 3.0x, 10.9x and 57x faster per
  query on 20, 100 and 400 beam elements (errors 0.81%, 0.83%, 4.1%), breaking even after 1 to 2 queries;
  Galerkin on a reduced basis only 3x to 5x because it still evaluates the full-order force; the polynomial
  (EED) force 168x faster per query at 400 elements but with 18.7 s offline, 13.7% error and a failed Newton
  solve on the 100-element mesh. See the user guide for the full table.

### Changed (renames)
- Renamed modules and tests that carried wave or phase numbers: `rom_engine.wave13_validation` is now
  `rom_engine.cross_configuration_validation` (no alias kept; update imports); tests
  `test_wave13_validation`, `test_wave1_newton_robustness`, `test_shell4director_phase2/3` and the script
  `validate_phase_b_coupling.py` and doc `phase4_error_bounds_greedy_roadmap.md` now have descriptive names.

### Added (fea_engine)
- Batched stress recovery: `stress`, `strain` and `von_mises` evaluate each block with array operations
  (`vectorized=True`, default). Measured speed-up over the per-element loop (best of repeats, 2-core CPU,
  `paper/experiments/e2_recovery_speedup.py`): about 25x to 55x on Quad4 meshes of 100 to 3600 elements and
  about 35x to 45x on Hex8 meshes of 216 to 2744 elements (run-to-run variation a few percent); 10,000 Quad4
  elements take about 45 ms. `vectorized=False` keeps the old loop, which is also used
  automatically for blocks with position-dependent coefficients. Results agree to rounding.
- Optional torch paths (torch is not required): `stress/strain/von_mises(..., backend="torch", device=)`,
  differentiable `FESystem.stress_tensor(U)` / `von_mises_tensor(U)` that return torch tensors backpropagating to
  `U`, `ReducedSystem.solve(backend="torch", device=, method="auto"|"dense"|"cg")` (works with constraints) and
  `ReducedSystem.to_torch()`. `expand/recover/restrict/residual` accept torch tensors. Computation is float64.
  Tests: `tests/test_torch_recovery_and_linear_system.py` (torch cases skip when torch is not installed).
- `FESystem.form_linear_system(F=None, K=None)` returns a `ReducedSystem`: the constrained `K`, `F` (prescribed
  values already moved to the right-hand side), `restrict`, `expand`, `recover` (full `FEField`),
  `reduce_matrix(M)`, `solve()` and `residual()`. Use it to bring your own solver, preconditioner or ROM.
- `fea_engine.export.write_vtu` and `write_series`: ParaView/VisIt output (`.vtu`, `.pvd` time series) of meshes
  and `FEField` / `FieldSeries` results (vectors, rotations, mode shapes, complex fields, extra point and
  cell data; ASCII or base64 binary). NumPy only.

- Derived results as `FEField`s: `FESystem.stress`, `strain`, `von_mises` and `reactions`. Stress is computed
  at the integration points with the assembled `D` (coefficients included) and projected to the nodes;
  `at="elements"` gives element averages. Supported: Quad4/Quad8/Tri3/Tri6 plane stress and
  Hex8/Hex20/Tet4/Tet10 solids. `reactions` returns `K U - F` at constrained DOFs, with force and moment units.
- `fea_engine.coefficients`: material, density and thickness that vary in space. `by_position(fn, at=)`,
  `by_element(values)` and `from_material(fn, builder, at=)` are accepted by `assemble_stiffness`,
  `assemble_mass` and `assemble_lumped_mass`. `at="centroid"` (default, any element) is piecewise constant;
  `at="gauss"` evaluates at integration points for the supported plane-stress and solid elements.
  A constant coefficient reproduces the plain assembly. Linear assembly only; not with `vectorized=True`.
- `FEField` accepts per-DOF unit labels (`dof_units`), so stress fields report `Pa` and moments `N*m`.

- Springs, foundations and constraints (`fea_engine.boundary`): `add_spring` (grounded springs, optional
  support displacement), `add_elastic_foundation` (Winkler / Robin term `k (u - u_ref)` on boundary edges or
  faces, `k` constant or a function of position), `add_constraint` (general `sum c_i u_i = value`) and `tie`
  (equal-DOF ties between node sets, with an offset for periodic boundaries). Springs and foundations live in
  `K`, survive re-assembly and work in the nonlinear drivers. Constraints are eliminated exactly (the reduced
  matrix stays symmetric positive definite); `solve_static`, `solve_modal` and `form_linear_system` honour
  them, every other solver raises `NotImplementedError` instead of ignoring them.
- `fea_engine.convergence` (`run_study`, `observed_order`, `pairwise_orders`, `richardson`) for mesh-convergence
  studies, and an independent benchmark suite (`tests/test_benchmarks_convergence.py`): patch tests on all eight
  continuum elements, exact pure-bending reproduction by Quad8/Tri6, observed convergence orders for Quad4, Tri3,
  Euler-Bernoulli and Reissner beams and axial-bar vibration, and the large-deflection cantilever elastica checked
  against a SciPy quadrature.

### Changed (fea_engine)
- `FEField.magnitude()` raises a clear `ValueError` for a field with no translational components
  (previously it returned zeros); `FEField.plot()` of a scalar field draws its single component.
- The vtk export writes one-DOF fields as scalars.

## fea_engine 1.0.1

### Changed (behaviour you may notice)
- **Re-assembly replaces instead of accumulating.** Calling `assemble_stiffness`, `assemble_mass` or
  `assemble_lumped_mass` a second time used to silently add to the existing matrix (doubling it). It now replaces
  the matrix and emits a one-time `UserWarning`. Pass `accumulate=True` to add on purpose. A failed re-assembly
  restores the previous matrix.
- **Invalid input now fails loudly.** `fix_dofs`, `add_nodal_force`, `add_consistent_edge_load` and
  `add_consistent_facet_load` raise `ValueError` for an empty node selection, a node id outside the mesh, or a
  local DOF outside `0..dofs_per_node-1`. A rejected call leaves no partial constraints.
- **Solve guards.** `solve_static` raises `RuntimeError` if the stiffness was never assembled and warns when the
  load vector is all zeros with nothing prescribed. `solve_modal`, the transient solvers and `solve_harmonic`
  raise `RuntimeError` (previously `assert`) when `M` is missing.
- **Scalar density.** A scalar `rho` given to a plane or solid element raises a `ValueError` that says to pass
  `rho * np.eye(n)`.
- **Result type.** `solve_static`, `solve_modal` (mode shapes) and `solve_harmonic` return `FEField`, an
  `ndarray` subclass. Indexing, arithmetic, reductions and saving behave as before.

### Added
- Named DOFs on every element (`dof_names`, aliases) accepted by `fix_dofs` and the load methods;
  `FESystem.dof_names`, `FESystem.dof_index`.
- Named node sets: `Mesh.add_node_set`, `node_set`, `select_nodes(x=, y=, z=, name=)`, usable wherever node ids are.
- `FEField` with `.component`, `.nodal`, `.at`, `.magnitude`, `.to_dataframe`, `.plot`, `.unit_of`; `FESystem.field`.
- `FieldSeries` for load paths, time histories and modes; `FESystem.series`, `FESystem.modal_series`.
- `NewtonOptions`: shared `tol`, `max_iter`, `du_tol`, `energy_tol`, `line_search`, `verbose`; every nonlinear and
  contact driver accepts `options=`. Explicit keywords win.
- `fea_engine.batch.map` for parameter sweeps (serial, thread or process pools, ordered results, error collection).
- `fea_engine.units`: unit labels (`SI`, `MM_N_TONNE`, `UnitSystem`), `Material.units`, `FESystem.units`.
  Labels only; nothing is converted.
- MIT `LICENSE` in each package, author and SPDX headers in every module.

### Fixed
- Invalid-escape `DeprecationWarning` in `tests/test_nonlinear_cantilever.py`; the intended zero-load warning in
  `tests/test_adaptivity.py` is filtered.

## rom_engine 0.2.0

### Added
- `cms` (Craig-Bampton, Guyan, substructure assembly), `hyper_reduction` (ECSW, DEIM/QDEIM, gappy POD, hooked into
  `IntrusiveNonlinearROM`), `linear_dynamics` (reduced Newmark, modal superposition), `validation` (`validate_rom`).
- MIT `LICENSE`, author and SPDX headers in every module.

### Fixed
- Balanced truncation, Hankel-norm and frequency-weighted truncation were sensitive to the BLAS/LAPACK build
  (4 test failures with Intel MKL). The state-space system is now equilibrated with power-of-two diagonal scaling
  (LAPACK `?GEBAL`) before the Gramians are computed.

## Earlier versions
`fea_engine` 1.0.0 and `rom_engine` 0.1.0 were the first tagged states of the packages.
