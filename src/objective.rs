//! The `Objective` trait: a typed function `Obj : S -> R` from the IISE manuscript.

use ndarray::{Array1, ArrayView1, ArrayView2};
use num_traits::Float;
use crate::{Bounds, FPair};

/// A real-valued function on `R^dim` with a known feasible domain.
///
/// The IISE manuscript's `Obj` signature: a typed map from a state space
/// `S = R^dim` (with `Bounds`) to `R`. Implementors may override
/// `eval_batch` and `global_min` for performance or known-optima
/// instrumentation.
pub trait Objective<T: Float>: Send + Sync {
    /// Number of input dimensions; matches `bounds().dims`.
    fn dim(&self) -> usize;

    /// The feasible domain (typically a box).
    fn bounds(&self) -> &Bounds<T>;

    /// Evaluates the objective at a single point.
    fn eval(&self, x: ArrayView1<T>) -> T;

    /// Evaluates at a batch of points; default implementation iterates `eval`.
    fn eval_batch(&self, x: ArrayView2<T>) -> Array1<T> {
        x.outer_iter().map(|row| self.eval(row)).collect()
    }

    /// Optional known global minimum, used for benchmarking and convergence
    /// checks. Default returns `None`.
    fn global_min(&self) -> Option<&FPair<T>> {
        None
    }
}
