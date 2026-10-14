# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
rom_engine -- a small, from-scratch reduced-order-modeling package for
structural engineering analysis.

This is the companion package to fea_engine: fea_engine assembles and
solves full-order finite element models, rom_engine compresses and
accelerates them. The two are deliberately DECOUPLED -- rom_engine's
core modules take plain numpy arrays (K, M, F, snapshot matrices), not
fea_engine objects, so this package works with matrices from any FE
source, not just fea_engine. fea_engine models are used as the
VALIDATION fixtures in this package's own test suite (see tests/), and
as the worked examples in examples/, but the library code itself never
imports fea_engine.

Modules (each module docstring has the full description, methods and limitations)
----------------------------------------------------------------------
affine                         : affine parametric decomposition, the "offline-online" trick that makes reduced-
                                 basis methods fast for many-query problems (parameter sweeps, optimization
                                 loops, Monte Carlo / uncertainty quantification -- anything that needs the SAME
                                 structure re-solved at many different parameter values).

attractor                      : steady-state, stroboscopic and attractor diagnostics for forced nonlinear
                                 structures: samples trajectories of full-order and reduced
                                 (IntrusiveNonlinearROM) models at the forcing period, measures steady-state
                                 amplitudes, sweeps frequency or amplitude, and extracts slaved-mode relations
                                 from POD amplitude series.

balanced_truncation            : balanced truncation (Besselink et al. 2013, Section 4 / eqs. 37-58), the
                                 "systems and control" family of model reduction.

cms                            : component mode synthesis: Guyan (static) condensation and Craig-Bampton
                                 substructure reduction, plus coupling of reduced substructures into one reduced
                                 model.

complementarity                : small dense linear complementarity (LCP) solvers and the reduced unilateral-
                                 contact solve.

component_pod                  : component-wise POD reduction of non-matching assemblies coupled through a
                                 mortar operator.

contact_comparison             : one call that runs several reduced contact models on the same held-out inputs
                                 and reports displacement, gap, contact-force and complementarity errors, size,
                                 selected elements, online time and the effective speed-up that charges offline
                                 cost against the number of queries.

contact_dual_basis             : reduced unilateral contact with a POSITIVE basis for the contact forces.

contact_greedy                 : weak-greedy snapshot selection for parametric contact ROMs, driven by the
                                 violation of the contact conditions (Balajewicz, Amsallem and Farhat 2015,
                                 Section 4.5).

contact_hyper_reduction        : hyper-reduction for unilateral contact problems.

contact_rid                    : hybrid hyper-reduction of unilateral contact on a Reduced Integration Domain
                                 (RID).

contact_validation             : error metrics for reduced contact models (plain arrays).

cross_configuration_validation : a cross-configuration validation harness that checks the excitation-response
                                 operator (`neural_operator.reduced_eom_residual_trajectory`), the regularised
                                 projection encoder (`reduced_basis_operator.RegularizedProjectionEncoder`) and
                                 the parameterised latent ODE
                                 (`parameterized_latent_ode.integrate_rk4`/`ParameterizedLatentODE`) together on
                                 ONE real `fea_engine` fixture pair, rather than validating each in isolation.

dataset_diagnostics            : a small, standalone utility: given a design-of-experiments parameter table (for
                                 example `rom_engine.sampling.optimal_lhs()`'s output, rescaled to physical
                                 units) and the resulting converged-case outputs, report.

differentiable_correction      : a design note and prototype (not a full reimplementation) for calibrating a
                                 trainable correction of a reduced model by explicit residual minimisation: the
                                 residual-minimisation calibration pattern of `fea_engine.differentiable`,
                                 applied to rom_engine's own reduced equilibrium equation instead of a full
                                 finite-element residual.

domain_partition               : linear / nonlinear domain partitioning with static condensation of the linear
                                 zone.

dual_modes                     : dual modes, the non-intrusive way to give a modal basis the membrane (in-plane)
                                 freedom that vibration modes of a thin structure lack.

ensemble_uq                    : ensemble uncertainty quantification for trainable corrections: train an NN or
                                 RBF correction N times from different random seeds or mini-batch orderings and
                                 report the ensemble mean and standard deviation of its predictions, a cheap,
                                 well-understood epistemic-uncertainty proxy.

forced_response_hbm            : forced, damped multi-harmonic response of a reduced model with a local
                                 nonlinearity, traced over excitation frequency by pseudo-arclength
                                 continuation.

frequency                      : intrusive frequency-domain reduced-order modeling for linear structural
                                 dynamics.

friction_contact_models        : dry-friction contact elements for harmonic balance (time-domain evaluation).

galerkin                       : intrusive Galerkin projection: given a reduced basis V (from pod.py, or an
                                 ordinary eigenbasis, or any other orthonormal-ish set of column vectors),
                                 project a full-order linear system onto it and solve the tiny reduced system
                                 instead.

greedy                         : weak-greedy reduced-basis training for frequency.FrequencyROM: instead of
                                 training a POD-on-FRF-snapshots basis from a fixed (e.g. uniform) grid of
                                 frequencies, ADAPTIVELY pick which frequencies to full-order-solve at, spending
                                 that (expensive) budget where the current basis is actually weak.

hankel_norm                    : optimal Hankel norm approximation (Adamjan-Arov-Krein theory; explicit state-
                                 space realisation due to Glover, "All optimal Hankel-norm approximations of
                                 linear multivariable systems and their L-infinity error bounds", 1984), an
                                 extension of the balanced-truncation family. The derivation and the numerical-risk investigation are summarised in
                                 docstring summarises.

hyper_reduced_hbm              : ECSW hyper-reduction of the interface elements of a jointed structure in
                                 harmonic balance, and a residual-based error indicator that needs no full-order
                                 solution (after Morsy, Kast and Tiso, 2022, arXiv:2204.12160, Sections
                                 3.3-3.5).

hyper_reduction                : hyper-reduction for nonlinear ROMs: ECSW, DEIM/QDEIM, gappy reconstruction.

interface_reduction            : interface-reduction and dual-assembly variants of component mode synthesis, for
                                 substructures that meet a joint, built on `cms.py`.

intrusive_nonlinear_rom        : an INTRUSIVE nonlinear Galerkin ROM that keeps the reduced
                                 mass/damping/stiffness matrices FULL (not mass-normalised, not diagonal): POD-
                                 Galerkin reduction of a nonlinear structure's equations of motion, with the
                                 full-order internal force evaluated at every query.

krylov                         : Krylov-subspace moment-matching ROM (Besselink et al. 2013, Section 3 / eqs.
                                 21-36), the "numerical mathematics" family of model reduction.

linear_dynamics                : time-domain response of LINEAR (full or reduced) structural systems.

loewner                        : non-intrusive modal parameter identification via the Loewner-pencil framework.

membrane_expansion             : membrane-basis estimation and expansion for implicit condensation and expansion
                                 (ICE) ROMs (Hollkamp and Gordon 2008, Sec. 2.2, Eqs. 9-16, Table 1 steps 6-8),
                                 which `nonlinear_rom.py` does not provide (its module docstring flags: "(e.g.
                                 membrane augmentation) this module does not yet implement").

metrics                        : generic, method-agnostic comparison metrics.

modal_derivatives              : modal derivatives and their selection and use as a basis augmentation, after
                                 Idelsohn and Cardona (1985), Weeger, Wever and Simeon (2016), Wu and Tiso
                                 (2016), Jain, Tiso, Rutzmoser and Rixen (2017), and the overview in Tiso,
                                 Karamooz Mahdiabadi and Marconi (2021), Section 4.5.2.

mode_correction                : mode acceleration and modal truncation augmentation (Besselink, Tabak,
                                 Lutowska, van de Wouw, Nijmeijer, Rixen, Hochstenbach & Schilders, JSV 2013,
                                 Section 2 / eqs. 13-20): two cheap, closed-form corrections to plain mode
                                 displacement (x ~= V @ eta, exactly what galerkin.GalerkinROM already gives
                                 from a modal basis) that recover the STATIC contribution of whatever modes a
                                 truncated basis V leaves out.

neural_operator                : an excitation-to-response operator for NONLINEAR reduced dynamics, following
                                 the architectural pattern of FRINO (Yang et al. 2025, "FE reduced-order model-
                                 informed neural operator for structural dynamic response prediction"): train
                                 ONE operator mapping a reduced excitation time series directly to a predicted
                                 response trajectory, so a single trained model generalises to excitation
                                 WAVEFORMS it never saw in training.

nnm                            : generalized multi-harmonic-balance (HBM) continuation for nonlinear normal mode
                                 (NNM) backbone curves.

nonlinear_dynamics             : generalized reduced nonlinear time integration.

nonlinear_rom                  : shared regression protocol for reduced nonlinear- force models, plus two
                                 concrete model families.

operator_inference             : second-order operator inference with Craig-Bampton structure, for contact
                                 dynamics.

parameterized_latent_ode       : a parameterised latent-ODE model, `dz/dt = f(t, xi, z)`, trained across a SWEEP
                                 of a physical parameter `xi` held fixed per trajectory (a load amplitude, a
                                 geometric or material property, ...) by curriculum learning.

parametric_modes               : first-order sensitivity of vibration modes to a parameter, for example a shape
                                 imperfection amplitude, a thickness or a material constant.

passivity                      : passivity (positive-realness) diagnostics for a collocated force-in/velocity-
                                 out structural port.

pod                            : Proper Orthogonal Decomposition (POD) basis extraction.

quadratic_manifold             : a quadratic manifold for geometrically nonlinear reduced-order models, its
                                 hyper-reduction with ECSW, and simulation-free training.

random_vibration               : PSD (power spectral density) random-vibration response, a thin layer on top of
                                 frequency.FrequencyROM's sweep.

reduced_basis_operator         : a discretisation-robust encoder/decoder pair in the style of reduced-order
                                 neural operators (RONOM, Dummer, Ye and Brune): encode a field sampled at ANY
                                 set of points into a finite-dimensional latent code by a regularised L2
                                 projection onto a CONTINUOUS basis, then decode that code to a field value at
                                 ANY OTHER set of points, including points never seen during fitting.

rubin_dual_basis               : Rubin free-interface basis enriched with dual modes, for structures with a
                                 (frictional) contact patch, and re-identification of the reduced polynomial
                                 force from a reduced time history (after Mashayekhi and Zucca, 2026).

sampling                       : generalized Optimal Latin Hypercube Sampling (OLHS) design-of-experiments core.

scm                            : a rigorous (provable, not just a practical indicator) lower bound on
                                 sigma_min(A(omega)), the quantity that turns FrequencyROM.residual_norm() into
                                 an a posteriori error BOUND rather than an estimate: ||x_true(omega) -
                                 x_ROM(omega)|| <= residual_norm(omega) / sigma_min(A(omega)).

scm_lp                         : the classical Successive Constraint Method (SCM; Huynh, Rozza, Sen, Patera
                                 2007): a rigorous lower bound on sigma_min(A(omega)) computed via a small
                                 ONLINE LINEAR PROGRAM over offline-selected reference ("control") points,
                                 rather than scm.py's simplified single-Lipschitz-constant construction.

screening                      : automated cross-ROM stability screening for non-intrusive Loewner-pencil modal
                                 identification.

slow_fast                      : slow-fast static condensation of the in-plane (membrane) coordinates of a
                                 geometrically nonlinear thin structure against its bending coordinates.

soar                           : SOAR (Second-Order Arnoldi) moment-matching ROM: a second-order-structure-
                                 preserving alternative to krylov.py's first-order Krylov moment matching.

state_space                    : convert a second-order structural system (M, C, K), plus an explicit INPUT map
                                 B and OUTPUT map Cout, into first-order state-space form.

subspace_nnm                   : nonlinear normal mode (NNM) backbone of a reduced model built in an augmented
                                 subspace (vibration modes plus modal derivatives or dual modes).

torch_contact_bases            : opt-in PyTorch counterparts of the dense, differentiable parts of the contact
                                 modules: the non-negative dual basis (`contact_dual_basis.nonneg_dual_basis`),
                                 the SPD operator fit of `operator_inference.fit_spd_operators`, the joint
                                 interface modes and Gram-Schmidt interface modes of `interface_reduction`, and
                                 `rubin_dual_basis.reidentify_polynomial_force`.

torch_linalg                   : shared, backend-neutral torch-native dense linear-algebra primitives for
                                 rom_engine's classical state-space MOR family (balanced_truncation.py,
                                 hankel_norm.py; see those modules' docstrings for which of their functions
                                 route through this one).

torch_system_bases             : opt-in PyTorch counterparts of the system-driven-basis and quadratic-manifold
                                 modules (`modal_derivatives`, `dual_modes`, `quadratic_manifold`, `slow_fast`,
                                 the free-interface part of `cms`, and `parametric_modes`), placed SIDE BY SIDE
                                 with the NumPy/SciPy originals rather than replacing them.

validation                     : one standard way to check ANY reduced model against its full-order model.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
from .pod import PodBasis, MultiFieldPOD, assemble_field_weight_matrix, trapezoidal_field_gram
from .galerkin import GalerkinROM
from .affine import AffineDecomposition
from .frequency import FrequencyROM, build_pod_basis_from_frf_snapshots
from .greedy import greedy_train_frequency_basis
from .random_vibration import psd_response, band_limited_gaussian_time_history
from .scm import SingularValueLowerBound, certified_error_bound
from .scm_lp import LPSingularValueLowerBound, certified_error_bound as certified_error_bound_lp
from .metrics import modal_assurance_criterion, r_squared
from .loewner import LoewnerROM
from .screening import screen_physical_modes, ScreenedMode
from .sampling import optimal_lhs, modal_force_samples
from .nonlinear_rom import (
    KERNEL_REGISTRY, ReducedForceModel, MultiFidelitySurrogate, PolynomialModalROM,
    NeuralSurrogate,
    TrainingStrategy, AppliedLoadStrategy, EnforcedDisplacementStrategy,
    TrajectoryPilotedStrategy,
    ICEROM, ShiMeiROM, EnforcedDisplacementROM,
)
from .nonlinear_dynamics import integrate_newmark_surrogate
from .nnm import HarmonicBalanceSystem, solve_nnm_backbone, solve_nnm_backbone_arclength
from .state_space import to_state_space, StateSpaceSystem
from .mode_correction import (
    mode_acceleration_correction, mode_acceleration_response, augmented_basis,
)
from .krylov import arnoldi_basis, two_sided_arnoldi_bases, KrylovROM
from .soar import soar_basis, SOARROM
from .balanced_truncation import (
    controllability_gramian, observability_gramian, hankel_singular_values,
    BalancedTruncationROM, SingularPerturbationROM,
    frequency_weighted_gramians, frequency_weighted_hankel_singular_values,
    FrequencyWeightedBalancedTruncationROM, lowpass_weight, bandpass_weight,
)
from .passivity import velocity_transfer_function, passivity_margin, is_passive
from .hankel_norm import OptimalHankelNormROM
from .dataset_diagnostics import (
    parameter_coverage_report, pca_dimensionality, GPSurrogate, sobol_indices,
)
from .ensemble_uq import EnsembleUQ, neural_surrogate_ensemble
from .differentiable_correction import (
    reduced_residual, ScalarModalCorrection, calibrate_reduced_correction_explicit,
)
from .membrane_expansion import MembraneBasis
from .neural_operator import reduced_eom_residual_trajectory
from .reduced_basis_operator import RegularizedProjectionEncoder
from .parameterized_latent_ode import rk4_step, integrate_rk4, CurriculumSchedule
from .cross_configuration_validation import (
    cross_check_reduced_dynamics, cross_check_basis_generalization,
    run_cross_configuration_validation,
)
from .intrusive_nonlinear_rom import (
    IntrusiveNonlinearROM, impulse_to_reduced_velocity,
    pod_rm_analysis, PodRMAnalysis, MatrixSPDReport,
)
from .attractor import (
    stroboscopic_sample, steady_state_amplitude, midspan_transverse_probe,
    master_slave_data, dominant_frequency,
    SweepPoint, SweepResult, frequency_sweep, amplitude_sweep,
)
# The torch-gated classes (ExcitationResponseOperator,
# LatentRefinementNet, ParameterizedODEFunc, ParameterizedLatentODE,
# cross_configuration_validation.cross_check_operators_torch) genuinely subclass
# torch.nn.Module and so, unlike NeuralSurrogate above, cannot be
# DEFINED at all without torch importable -- deliberately NOT
# re-exported here (see neural_operator's own module docstring note);
# import them directly from their own module when torch is installed.

from .cms import (guyan, craig_bampton, couple, GuyanModel, CraigBamptonModel, CoupledModel,
                  free_interface_cms, reduce_interface, FreeInterfaceModel, InterfaceReducedModel)
from .hyper_reduction import (
    deim_indices, qdeim_indices, DEIM, gappy_reconstruct, ecsw_weights,
    ecsw_training_matrix, ECSW, reduced_force_error, HyperReducedNonlinearROM,
)
from .linear_dynamics import newmark_linear, modal_superposition, piecewise_linear_exact, galerkin_transient
from .validation import (validate_rom, convergence_study, select_basis_size, ValidationReport,
                         effective_speedup, break_even_queries, compare_bases)
from .modal_derivatives import (
    directional_stiffness_derivative, ModalDerivatives, static_modal_derivatives, dynamic_modal_derivatives,
    modal_coordinates, modal_interaction_weights, select_modal_derivatives, augment_with_derivatives,
)
from .dual_modes import dual_modes, amplitude_to_load_factor
from .nonlinear_rom import enhanced_enforced_displacement_rom, EnhancedEnforcedDisplacementROM
from .quadratic_manifold import (
    QuadraticManifold, fit_quadratic_manifold, manifold_from_derivatives, lifted_training_states,
    ManifoldECSW, QuadraticManifoldROM,
)
from .slow_fast import SlowFastROM
from .subspace_nnm import diagonalize_basis, nnm_backbone_in_subspace
from .parametric_modes import mode_sensitivities, updated_modes

from .complementarity import lcp_lemke, lcp_active_set, solve_lcp, solve_reduced_contact, complementarity_violation
from .contact_dual_basis import (nonneg_dual_basis, svd_dual_basis, active_snapshot_columns, ContactROM,
                                 ContactROMResult, NMFBasis, sparse_dictionary_fit)
from .contact_greedy import contact_condition_indicator, GreedyContactHistory, greedy_contact_sampling
from .contact_rid import (pod_by_information_rate, rid_elements, rid_dofs, contact_rows_in_rid, HybridHRContact,
                          HybridHRResult, restricted_fom_solve, snapshot_interpolator, h1_norm, dual_h1_norm,
                          interpolation_indicator, deim_rid_indices)
from .contact_hyper_reduction import (ecsw_weights_with_required, fit_contact_aware_ecsw, complete_constraint_elements,
                                      reconstruct_contact_forces, rank_deficiency, enrich_basis_for_constraints,
                                      solve_reduced_contact_newton, ReducedContactNewtonResult)
from .domain_partition import nonlinearity_indicator, select_nonlinear_elements, PartitionedModel
from .contact_validation import max_normalised_error, contact_errors, ContactErrors, contact_set_agreement

from .friction_contact_models import JenkinsContact, jenkins_hysteresis_energy
from .forced_response_hbm import (ForcedHBM, PolynomialForce, ForcedResponse, forced_response_continuation,
                                  equivalent_resonance, equivalent_properties_vs_force)
from .interface_reduction import (DualCraigBamptonModel, dual_craig_bampton, couple_dual, joint_interface_modes,
                                  jim_craig_bampton, gram_schmidt_interface_modes, interface_reduced_craig_bampton,
                                  fourier_interface_basis, multistage_fourier_basis, trial_vector_derivative_basis,
                                  natural_frequency_errors, mac_matrix, mac_diagonal, InterfaceReducedBasis)
from .component_pod import (legendre_interface_parametrisation, component_snapshots, ComponentBasis, ReducedAssembly,
                            ReducedAssemblyResult)
from .operator_inference import (static_coupling, lstsq_coupling, craig_bampton_transformation, fit_spd_operators,
                                 OperatorInferenceModel, infer_contact_operators)
from .hyper_reduced_hbm import (JointElementSet, harmonic_training_matrix, fit_harmonic_ecsw, HyperReducedJointForce,
                                residual_error_indicator)
from .rubin_dual_basis import (rubin_dual_basis, select_by_representation_error, physical_contact_basis,
                               PolynomialForceFit, reidentify_polynomial_force)
from .contact_comparison import compare_contact_models, format_contact_comparison

__all__ = [
    "PodBasis", "MultiFieldPOD", "assemble_field_weight_matrix", "trapezoidal_field_gram",
    "GalerkinROM", "AffineDecomposition",
    "FrequencyROM", "build_pod_basis_from_frf_snapshots",
    "greedy_train_frequency_basis",
    "psd_response", "band_limited_gaussian_time_history",
    "SingularValueLowerBound", "certified_error_bound",
    "LPSingularValueLowerBound", "certified_error_bound_lp",
    "modal_assurance_criterion", "r_squared",
    "LoewnerROM",
    "screen_physical_modes", "ScreenedMode",
    "optimal_lhs", "modal_force_samples",
    "KERNEL_REGISTRY", "ReducedForceModel", "MultiFidelitySurrogate", "PolynomialModalROM",
    "NeuralSurrogate",
    "TrainingStrategy", "AppliedLoadStrategy", "EnforcedDisplacementStrategy",
    "TrajectoryPilotedStrategy",
    "ICEROM", "ShiMeiROM", "EnforcedDisplacementROM",
    "integrate_newmark_surrogate",
    "HarmonicBalanceSystem", "solve_nnm_backbone", "solve_nnm_backbone_arclength",
    "to_state_space", "StateSpaceSystem",
    "mode_acceleration_correction", "mode_acceleration_response", "augmented_basis",
    "arnoldi_basis", "two_sided_arnoldi_bases", "KrylovROM",
    "soar_basis", "SOARROM",
    "controllability_gramian", "observability_gramian", "hankel_singular_values",
    "BalancedTruncationROM", "SingularPerturbationROM",
    "frequency_weighted_gramians", "frequency_weighted_hankel_singular_values",
    "FrequencyWeightedBalancedTruncationROM", "lowpass_weight", "bandpass_weight",
    "velocity_transfer_function", "passivity_margin", "is_passive",
    "OptimalHankelNormROM",
    "parameter_coverage_report", "pca_dimensionality", "GPSurrogate", "sobol_indices",
    "EnsembleUQ", "neural_surrogate_ensemble",
    "reduced_residual", "ScalarModalCorrection", "calibrate_reduced_correction_explicit",
    "MembraneBasis",
    "reduced_eom_residual_trajectory",
    "RegularizedProjectionEncoder",
    "rk4_step", "integrate_rk4", "CurriculumSchedule",
    "cross_check_reduced_dynamics", "cross_check_basis_generalization",
    "run_cross_configuration_validation",
    "IntrusiveNonlinearROM", "impulse_to_reduced_velocity",
    "pod_rm_analysis", "PodRMAnalysis", "MatrixSPDReport",
    "stroboscopic_sample", "steady_state_amplitude", "midspan_transverse_probe",
    "master_slave_data", "dominant_frequency",
    "SweepPoint", "SweepResult", "frequency_sweep", "amplitude_sweep",
    "guyan", "craig_bampton", "couple", "GuyanModel", "CraigBamptonModel", "CoupledModel",
    "deim_indices", "qdeim_indices", "DEIM", "gappy_reconstruct", "ecsw_weights",
    "ecsw_training_matrix", "ECSW", "reduced_force_error", "HyperReducedNonlinearROM",
    "newmark_linear", "modal_superposition", "piecewise_linear_exact", "galerkin_transient",
    "validate_rom", "convergence_study", "select_basis_size", "ValidationReport",
    "effective_speedup", "break_even_queries", "compare_bases",
    "free_interface_cms", "reduce_interface", "FreeInterfaceModel", "InterfaceReducedModel",
    "directional_stiffness_derivative", "ModalDerivatives", "static_modal_derivatives",
    "dynamic_modal_derivatives", "modal_coordinates", "modal_interaction_weights", "select_modal_derivatives",
    "augment_with_derivatives", "dual_modes", "amplitude_to_load_factor",
    "enhanced_enforced_displacement_rom", "EnhancedEnforcedDisplacementROM",
    "QuadraticManifold", "fit_quadratic_manifold", "manifold_from_derivatives", "lifted_training_states",
    "ManifoldECSW", "QuadraticManifoldROM", "SlowFastROM", "diagonalize_basis", "nnm_backbone_in_subspace",
    "mode_sensitivities", "updated_modes",
    "lcp_lemke", "lcp_active_set", "solve_lcp", "solve_reduced_contact", "complementarity_violation", "nonneg_dual_basis", "svd_dual_basis", "active_snapshot_columns", "ContactROM", "ContactROMResult", "NMFBasis", "sparse_dictionary_fit", "contact_condition_indicator", "GreedyContactHistory", "greedy_contact_sampling", "ecsw_weights_with_required", "fit_contact_aware_ecsw", "complete_constraint_elements", "reconstruct_contact_forces", "rank_deficiency", "enrich_basis_for_constraints", "solve_reduced_contact_newton", "ReducedContactNewtonResult", "nonlinearity_indicator", "select_nonlinear_elements", "PartitionedModel", "max_normalised_error", "contact_errors", "ContactErrors", "contact_set_agreement",
    "JenkinsContact", "jenkins_hysteresis_energy", "ForcedHBM", "PolynomialForce", "ForcedResponse", "forced_response_continuation", "equivalent_resonance", "equivalent_properties_vs_force", "DualCraigBamptonModel", "dual_craig_bampton", "couple_dual", "joint_interface_modes", "jim_craig_bampton", "gram_schmidt_interface_modes", "interface_reduced_craig_bampton", "fourier_interface_basis", "multistage_fourier_basis", "trial_vector_derivative_basis", "natural_frequency_errors", "mac_matrix", "mac_diagonal", "InterfaceReducedBasis", "legendre_interface_parametrisation", "component_snapshots", "ComponentBasis", "ReducedAssembly", "ReducedAssemblyResult", "static_coupling", "lstsq_coupling", "craig_bampton_transformation", "fit_spd_operators", "OperatorInferenceModel", "infer_contact_operators", "JointElementSet", "harmonic_training_matrix", "fit_harmonic_ecsw", "HyperReducedJointForce", "residual_error_indicator", "rubin_dual_basis", "select_by_representation_error", "physical_contact_basis", "PolynomialForceFit", "reidentify_polynomial_force", "compare_contact_models", "format_contact_comparison",
    "pod_by_information_rate", "rid_elements", "rid_dofs", "contact_rows_in_rid", "HybridHRContact", "HybridHRResult", "restricted_fom_solve", "snapshot_interpolator", "h1_norm", "dual_h1_norm", "interpolation_indicator", "deim_rid_indices",
]

__version__ = "0.2.0"
