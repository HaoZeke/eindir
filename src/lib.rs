//! eindir-core: typed primitives for ND objective functions.
//!
//! This crate ships the typed primitives (FPair, Bounds, Objective trait,
//! and builtin objective implementations) consumed by `anneal-core`. The
//! current revision is a structural scaffold; the typed component algebra
//! lands in v0.3.0.

pub mod error;
pub mod types;
#[cfg(feature = "capi")]
pub mod ffi;
#[cfg(feature = "python")]
pub mod python;

pub use error::Error;
