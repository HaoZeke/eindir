//! Forward-mode automatic differentiation via `num-dual`.
//!
//! Write the scalar field once, generic over [`num_dual::DualNum`]. This
//! module lifts it to [`Objective`] + [`Gradient`] + [`DifferentiableObjective`]
//! with a fused `value_and_gradient` (one dual evaluation, exact partials).
//! Central differences stay available as [`crate::FiniteDiffGradient`] when
//! the function cannot be instantiated at a dual type.

use crate::{Bounds, DifferentiableObjective, Gradient, Objective};
use nalgebra::DVector;
use ndarray::{Array1, ArrayView1};
use num_dual::{DualDVec64, DualNum, gradient};

/// A scalar field `R^n -> R` written generically over dual numbers.
pub trait DualField: Send + Sync {
    /// Evaluates `f` at `x`. Instantiated at `f64` for values and at
    /// [`DualDVec64`] for an exact gradient in one sweep.
    fn eval_dual<D: DualNum<f64> + Clone>(&self, x: &[D]) -> D;
}

/// Forward-mode AD wrapper around a [`DualField`].
pub struct ForwardAd<S: DualField> {
    field: S,
    bounds: Bounds<f64>,
}

impl<S: DualField> ForwardAd<S> {
    /// Wraps `field` on the given box.
    pub fn new(field: S, bounds: Bounds<f64>) -> Self {
        Self { field, bounds }
    }

    /// Exact `(f, ∇f)` at `x` from one dual evaluation.
    pub fn value_and_grad(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let xv = DVector::from_iterator(x.len(), x.iter().copied());
        let field = &self.field;
        let (val, g) = gradient(
            |v: DVector<DualDVec64>| {
                let buf: Vec<DualDVec64> = v.iter().cloned().collect();
                field.eval_dual(&buf)
            },
            &xv,
        );
        (val, Array1::from_iter(g.iter().copied()))
    }
}

impl<S: DualField> Objective<f64> for ForwardAd<S> {
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let buf: Vec<f64> = x.iter().copied().collect();
        self.field.eval_dual(&buf)
    }
}

impl<S: DualField> Gradient<f64> for ForwardAd<S> {
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.value_and_grad(x).1
    }
}

impl<S: DualField> DifferentiableObjective<f64> for ForwardAd<S> {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.value_and_grad(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objectives::Rosenbrock;
    use ndarray::array;

    struct RosenDual;

    impl DualField for RosenDual {
        fn eval_dual<D: DualNum<f64> + Clone>(&self, x: &[D]) -> D {
            let mut s = D::from(0.0);
            for i in 0..x.len().saturating_sub(1) {
                let xi = x[i].clone();
                let t = x[i + 1].clone() - xi.clone().powi(2);
                s = s + D::from(100.0) * t.powi(2) + (D::from(1.0) - xi).powi(2);
            }
            s
        }
    }

    #[test]
    fn dual_gradient_matches_closed_form_rosenbrock() {
        let bounds = Bounds::new(array![-2.048, -2.048], array![2.048, 2.048], 1e-9);
        let ad = ForwardAd::new(RosenDual, bounds);
        let x = array![-1.2, 1.0];
        let analytic = Rosenbrock::<2>::new();
        let (v_ad, g_ad) = ad.value_and_gradient(x.view());
        let (v, g) = analytic.value_and_gradient(x.view());
        assert!((v_ad - v).abs() < 1e-12);
        assert!((g_ad[0] - g[0]).abs() < 1e-12);
        assert!((g_ad[1] - g[1]).abs() < 1e-12);
    }
}
