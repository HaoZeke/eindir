//! The `Objective` trait: a typed function `Obj : S -> R`, the objective slot
//! of the anneal component algebra.

use crate::{Bounds, FPair};
use ndarray::{Array1, ArrayView1, ArrayView2};
use num_traits::Float;
use rayon::prelude::*;

/// Minimum batch size before the parallel `eval` fan-out is worth the spawn cost.
pub const EVAL_BATCH_PARALLEL_MIN: usize = 16;

/// A real-valued function on `R^dim` with a known feasible domain.
///
/// The `Obj` signature of the anneal component algebra: a typed map from a
/// state space `S = R^dim` (with `Bounds`) to `R`. Implementors may
/// override `eval_batch` for multi-walker or multi-start backends (Rayon,
/// one Python call per batch, worker pools) and `global_min` for
/// known-optima instrumentation.
///
/// `eval` cannot fail. A surface that can fail implements
/// [`crate::TryObjective`] instead, and every `Objective` is already a
/// `TryObjective` with `Error = Infallible`.
///
/// # Examples
///
/// ```
/// use eindir_core::{Bounds, Objective};
/// use ndarray::{ArrayView1, array};
///
/// struct Sphere(Bounds<f64>);
///
/// impl Objective<f64> for Sphere {
///     fn dim(&self) -> usize {
///         self.0.dims
///     }
///     fn bounds(&self) -> &Bounds<f64> {
///         &self.0
///     }
///     fn eval(&self, x: ArrayView1<f64>) -> f64 {
///         x.iter().map(|v| v * v).sum()
///     }
/// }
///
/// let s = Sphere(Bounds::new(array![-1.0, -1.0], array![1.0, 1.0], 0.0));
/// assert_eq!(s.eval(array![0.5, 0.5].view()), 0.5);
/// assert_eq!(s.eval_batch(array![[1.0, 0.0], [0.0, 0.0]].view()), array![1.0, 0.0]);
/// ```
pub trait Objective<T: Float>: Send + Sync {
    /// Number of input dimensions; matches `bounds().dims`.
    fn dim(&self) -> usize;

    /// The feasible domain (typically a box).
    fn bounds(&self) -> &Bounds<T>;

    /// Evaluates the objective at a single point.
    ///
    /// Callers pass `dim` coordinates. Nothing checks that `x` is inside
    /// `bounds()`; implementors decide what a point outside means.
    fn eval(&self, x: ArrayView1<T>) -> T;

    /// Evaluates a batch of points (rows of `x`).
    ///
    /// Default: serial loop over [`Objective::eval`]. Override for
    /// multi-walker fan-out (see [`eval_batch_parallel`] for native f64).
    fn eval_batch(&self, x: ArrayView2<T>) -> Array1<T> {
        x.outer_iter().map(|row| self.eval(row)).collect()
    }

    /// Optional known global minimum, used for benchmarking and convergence
    /// checks. Default returns `None`.
    fn global_min(&self) -> Option<&FPair<T>> {
        None
    }
}

/// Parallel multi-walker / multi-start batch evaluation for `Objective<f64>`.
///
/// Uses Rayon when `x.nrows() >= EVAL_BATCH_PARALLEL_MIN` and calls
/// [`Objective::eval_batch`] below that. The result is in row order and
/// does not depend on the thread count; `RAYON_NUM_THREADS` sets the pool
/// size. Prefer
/// [`Objective::eval_batch`] when the implementor has a specialized batch
/// path (Python/CUTEst); this helper is the native Sync hot path.
pub fn eval_batch_parallel<O>(obj: &O, x: ArrayView2<f64>) -> Array1<f64>
where
    O: Objective<f64> + ?Sized,
{
    let n = x.nrows();
    if n < EVAL_BATCH_PARALLEL_MIN {
        return obj.eval_batch(x);
    }
    let mut out = vec![0.0_f64; n];
    out.par_iter_mut().enumerate().for_each(|(i, slot)| {
        *slot = obj.eval(x.row(i));
    });
    Array1::from(out)
}
