#![warn(missing_docs)]
//! eindir-core: typed primitives for ND objective functions.
//!
//! This crate ships the typed primitives (FPair, Bounds, Objective trait,
//! gradients, surrogates, point sets, and builtin objective implementations)
//! consumed by `anneal-core`.

/// Separable rank-1 surrogate and its tempered independence sampler.
pub mod additive;
/// Box bounds on N-dimensional points.
pub mod bounds;
/// Error types shared by all fallible operations in `eindir-core`.
pub mod error;
/// Box geometry features and compensated ΔE.
pub mod features;
/// C ABI surface, gated behind the `capi` Cargo feature.
#[cfg(feature = "capi")]
#[allow(non_camel_case_types)]
pub mod ffi;
/// Position-value pairs.
pub mod fpair;
/// Generalized Langevin equation colored-noise thermostat (optimal sampling).
pub mod gle;
/// Gradient trait and adapters (Analytic / FiniteDiff) plus
/// `DifferentiableObjective`.  "Native" gradients when the caller can
/// supply them (Ceres-style); explicit central differences otherwise.
pub mod gradient;
/// The `Objective` trait: typed `S -> R` map with bounds and optional known minimum.
pub mod objective;
/// Builtin objective-function implementations for benchmarking and tests.
pub mod objectives;
/// Low-discrepancy point sets for bounded numerical objectives.
pub mod pointset;
/// Adapter wrapping a Python callable into the Rust `Objective<f64>` trait.
#[cfg(any(feature = "python", feature = "py-objective"))]
pub mod py_objective;
/// pyo3 bindings for `eindir._core`, gated behind the `python` Cargo feature.
#[cfg(feature = "python")]
pub mod python;
/// Dimension-collapse and Chebyshev surrogate `Obj`-transforms.
pub mod reduced;
/// Shared algebra-facing type definitions.
pub mod types;

pub use additive::AdditiveSurrogate;
pub use bounds::Bounds;
pub use error::Error;
pub use features::{BoxGeometry, box_geometry, compensated_delta, isotropic_proposal_scale};
pub use fpair::FPair;
pub use gle::{GleThermostat, OPTIMAL_SAMPLING_NS, ldl_sqrt, matrix_exp, optimal_sampling_drift};
pub use gradient::{AnalyticGradient, DifferentiableObjective, FiniteDiffGradient, Gradient};
pub use objective::{EVAL_BATCH_PARALLEL_MIN, Objective, eval_batch_parallel};
pub use objectives::{Ackley, Rastrigin, Rosenbrock, StybTang2D};
pub use pointset::{
    boundary_anchored_low_discrepancy_points, halton_points, halton_unit, low_discrepancy_points,
    radical_inverse, shifted_halton_points, shifted_low_discrepancy_points,
};
pub use reduced::{ChebyshevSurrogate, ReducedObjective};

#[cfg(feature = "capi")]
pub use ffi::{
    EindirEvalFn, EindirFreeFn, EindirGradFn, EindirObjectiveWrapper, eindir_abi_stamp_t,
    eindir_objective_t,
};
