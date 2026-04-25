#![warn(missing_docs)]
//! eindir-core: typed primitives for ND objective functions.
//!
//! This crate ships the typed primitives (FPair, Bounds, Objective trait,
//! and builtin objective implementations) consumed by `anneal-core`. The
//! current revision is a structural scaffold; the typed component algebra
//! lands in v0.3.0.

/// Error types shared by all fallible operations in `eindir-core`.
pub mod error;
/// Typed component algebra (placeholder; populated in v0.3.0).
pub mod types;
/// Position-value pairs.
pub mod fpair;
/// Box bounds on N-dimensional points.
pub mod bounds;
/// The `Objective` trait: typed `S -> R` map with bounds and optional known minimum.
pub mod objective;
/// Builtin objective-function implementations for benchmarking and tests.
pub mod objectives;
/// C ABI surface, gated behind the `capi` Cargo feature.
#[cfg(feature = "capi")]
pub mod ffi;
/// pyo3 bindings for `eindir._core`, gated behind the `python` Cargo feature.
#[cfg(feature = "python")]
pub mod python;

pub use error::Error;
pub use fpair::FPair;
pub use bounds::Bounds;
pub use objective::Objective;
pub use objectives::{StybTang2D, Rastrigin, Rosenbrock, Ackley};
