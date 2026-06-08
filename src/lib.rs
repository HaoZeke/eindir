#![warn(missing_docs)]
//! eindir-core: typed primitives for ND objective functions.
//!
//! This crate ships the typed primitives (FPair, Bounds, Objective trait,
//! and builtin objective implementations) consumed by `anneal-core`. The
//! current revision is a structural scaffold; the typed component algebra
//! lands in v0.3.0.

/// Box bounds on N-dimensional points.
pub mod bounds;
/// Error types shared by all fallible operations in `eindir-core`.
pub mod error;
/// C ABI surface, gated behind the `capi` Cargo feature.
#[cfg(feature = "capi")]
pub mod ffi;
/// Position-value pairs.
pub mod fpair;
/// The `Objective` trait: typed `S -> R` map with bounds and optional known minimum.
pub mod objective;
/// Builtin objective-function implementations for benchmarking and tests.
pub mod objectives;
/// Low-discrepancy point sets for bounded numerical objectives.
pub mod pointset;
/// Adapter wrapping a Python callable into the Rust `Objective<f64>` trait.
#[cfg(feature = "python")]
pub mod py_objective;
/// pyo3 bindings for `eindir._core`, gated behind the `python` Cargo feature.
#[cfg(feature = "python")]
pub mod python;
/// Dimension-collapse and Chebyshev surrogate `Obj`-transforms.
pub mod reduced;
/// Separable rank-1 surrogate and its tempered independence sampler.
pub mod additive;
/// Generalized Langevin equation colored-noise thermostat (optimal sampling).
pub mod gle;
/// Typed component algebra (placeholder; populated in v0.3.0).
pub mod types;

pub use bounds::Bounds;
pub use error::Error;
pub use fpair::FPair;
pub use objective::Objective;
pub use objectives::{Ackley, Rastrigin, Rosenbrock, StybTang2D};
pub use pointset::{
    boundary_anchored_low_discrepancy_points, halton_points, halton_unit,
    low_discrepancy_points, radical_inverse, shifted_halton_points,
    shifted_low_discrepancy_points,
};
pub use additive::AdditiveSurrogate;
pub use gle::{
    ldl_sqrt, matrix_exp, optimal_sampling_drift, GleThermostat, OPTIMAL_SAMPLING_NS,
};
pub use reduced::{ChebyshevSurrogate, ReducedObjective};
