//! `Gradient<T>` and `DifferentiableObjective<T>`: first-derivative support
//! for the typed `Obj` algebra.
//!
//! This is the "native gradient" path, modeled on how Ceres accepts analytic
//! derivatives: when a caller (HMC, projected-gradient polish, GLE with
//! optimal drift, NEB/saddle search in sibling crates, ...) can supply a
//! true derivative it is used directly; otherwise an explicit
//! `FiniteDiffGradient` adapter falls back to central differences (costing
//! 2·dim extra objective evaluations per gradient).
//!
//! The separation (value-only `Objective` vs. `Gradient` provider) mirrors
//! Stan's log-density / gradient split and Ceres' residual + optional
//! Jacobian in a single Evaluate.  A type can implement both (via
//! `DifferentiableObjective`) and supply an efficient fused
//! `value_and_gradient` to avoid redundant work.
//!
//! Surrogates (`ChebyshevSurrogate`, `AdditiveSurrogate`, `ReducedObjective`)
//! implement the traits analytically where possible.  Built-in test
//! objectives ship with closed-form gradients.  Python callables can
//! provide a matching `grad_fn` at construction time of `PyObjective`.

use crate::Objective;
use ndarray::{Array1, ArrayView1};
use num_traits::Float;

/// First-derivative interface.  Types that know an analytic gradient
/// implement this directly (or via `DifferentiableObjective`).
///
/// When only a black-box `Objective` is available, wrap it with
/// `FiniteDiffGradient`.  When a closure or Python grad callable is known,
/// use `AnalyticGradient` (or construct a `PyObjective` with `grad_fn`).
pub trait Gradient<T: Float>: Send + Sync {
    /// Gradient of the underlying scalar field at `x`.
    fn grad(&self, x: ArrayView1<T>) -> Array1<T>;

    /// Number of input dimensions (must match the paired `Objective::dim`).
    fn dim(&self) -> usize;
}

/// An `Objective` that can also supply its gradient (natively or via adapter).
///
/// There is no blanket impl: a type that is `Objective + Gradient` must
/// write `impl DifferentiableObjective<T> for T { }` (empty, default
/// split eval) or override [`DifferentiableObjective::value_and_gradient`]
/// for a fused `(f, ∇f)` that reuses intermediate work (Ceres-style
/// residual + Jacobian in one shot). A blanket impl would make that
/// override E0119.
pub trait DifferentiableObjective<T: Float>: Objective<T> + Gradient<T> {
    /// Returns `(f(x), ∇f(x))`.  Default calls the two methods separately;
    /// analytic implementations should override for efficiency.
    fn value_and_gradient(&self, x: ArrayView1<T>) -> (T, Array1<T>) {
        (self.eval(x), self.grad(x))
    }
}

/// Closure-based analytic gradient (the common case for user Rust code
/// that has a hand-derived or AD-produced `x |-> ∇f(x)`).
pub struct AnalyticGradient<F>
where
    F: Fn(ArrayView1<f64>) -> Array1<f64> + Send + Sync,
{
    /// User-supplied analytic gradient callback.
    pub grad_fn: F,
    /// Number of input dimensions accepted by `grad_fn`.
    pub dim: usize,
}

impl<F> AnalyticGradient<F>
where
    F: Fn(ArrayView1<f64>) -> Array1<f64> + Send + Sync,
{
    /// Constructs an analytic-gradient adapter from a dimension and callback.
    pub fn new(dim: usize, grad_fn: F) -> Self {
        Self { grad_fn, dim }
    }
}

impl<F> Gradient<f64> for AnalyticGradient<F>
where
    F: Fn(ArrayView1<f64>) -> Array1<f64> + Send + Sync,
{
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        (self.grad_fn)(x)
    }
    fn dim(&self) -> usize {
        self.dim
    }
}

/// Central-difference adapter around any `Objective<f64>`.
/// Use only when no native gradient is available.
pub struct FiniteDiffGradient<O: Objective<f64>> {
    /// Objective evaluated by the central-difference stencil.
    pub obj: O,
    /// Symmetric perturbation step.
    pub h: f64,
}

impl<O: Objective<f64>> FiniteDiffGradient<O> {
    /// Constructs a central-difference adapter with a conservative default step.
    pub fn new(obj: O) -> Self {
        Self { obj, h: 1e-5 }
    }
    /// Constructs a central-difference adapter with a caller-selected step.
    pub fn with_step(obj: O, h: f64) -> Self {
        assert!(h > 0.0, "h must be positive");
        Self { obj, h }
    }
}

impl<O: Objective<f64> + Send + Sync> Gradient<f64> for FiniteDiffGradient<O> {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let n = x.len();
        let mut g = Array1::zeros(n);
        let mut xp = x.to_owned();
        let mut xm = x.to_owned();
        for i in 0..n {
            xp[i] = x[i] + self.h;
            xm[i] = x[i] - self.h;
            g[i] = (self.obj.eval(xp.view()) - self.obj.eval(xm.view())) / (2.0 * self.h);
            xp[i] = x[i];
            xm[i] = x[i];
        }
        g
    }
    fn dim(&self) -> usize {
        self.obj.dim()
    }
}

#[cfg(test)]
mod fused_override_tests {
    use super::*;
    use crate::{Bounds, Objective};
    use ndarray::{Array1, ArrayView1};

    struct Counted;
    impl Objective<f64> for Counted {
        fn dim(&self) -> usize {
            1
        }
        fn bounds(&self) -> &Bounds<f64> {
            static CELL: std::sync::OnceLock<Bounds<f64>> = std::sync::OnceLock::new();
            CELL.get_or_init(|| Bounds::new(Array1::from_elem(1, -1.0), Array1::from_elem(1, 1.0), 0.0))
        }
        fn eval(&self, _x: ArrayView1<f64>) -> f64 {
            panic!("split eval must not run when value_and_gradient is fused")
        }
    }
    impl Gradient<f64> for Counted {
        fn dim(&self) -> usize {
            1
        }
        fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> {
            panic!("split grad must not run when value_and_gradient is fused")
        }
    }
    impl DifferentiableObjective<f64> for Counted {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (x[0] * x[0], Array1::from_elem(1, 2.0 * x[0]))
        }
    }

    #[test]
    fn override_is_fused() {
        let obj = Counted;
        let x = Array1::from_elem(1, 3.0);
        let (v, g) = obj.value_and_gradient(x.view());
        assert_eq!(v, 9.0);
        assert_eq!(g[0], 6.0);
    }
}
