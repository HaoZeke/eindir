//! Forward-mode automatic differentiation via `num-dual`.
//!
//! [`Dual64`] is one real value and one directional derivative. A map
//! `R^n -> R^m` written on it yields a Jacobian-vector product from
//! [`directional_derivative`], with no difference step.
//!
//! A scalar field is written once, generic over [`DualNum`]. [`ForwardAd`]
//! lifts that field to [`Objective`] + [`Gradient`] +
//! [`DifferentiableObjective`] with a fused `value_and_gradient` (one
//! [`num_dual::DualDVec64`] evaluation, the full gradient). Central
//! differences stay available as [`crate::FiniteDiffGradient`] when the
//! function cannot be instantiated at a dual type.

use crate::{Bounds, DifferentiableObjective, Error, Gradient, Objective};
use nalgebra::DVector;
use ndarray::{Array1, ArrayView1};
pub use num_dual::{Dual64, DualNum};
use num_dual::{DualDVec64, gradient};

/// Value of a map and its directional derivative along one seed.
#[derive(Clone, Debug, PartialEq)]
pub struct DirectionalDerivative {
    /// Image of the map at the seeded point.
    pub value: Vec<f64>,
    /// Jacobian of the map at that point, times the seed direction.
    pub derivative: Vec<f64>,
}

/// Jacobian-vector product of `map` along `direction`.
///
/// Each coordinate of `x` is seeded with the matching component of
/// `direction`. The real part of `map` is [`DirectionalDerivative::value`]
/// and the dual part is [`DirectionalDerivative::derivative`]. `map` chooses
/// the output length; it does not have to match `x.len()`.
///
/// `x` must be non-empty. Every component of `x`, `direction`, and both
/// parts of the image must be finite. An empty point or a non-finite value
/// is [`Error::OutOfBounds`]. A length mismatch is
/// [`Error::DimensionMismatch`]. This function does not catch a panic from
/// `map`.
pub fn directional_derivative<F>(
    x: &[f64],
    direction: &[f64],
    map: F,
) -> Result<DirectionalDerivative, Error>
where
    F: FnOnce(&[Dual64]) -> Vec<Dual64>,
{
    if x.is_empty() {
        return Err(Error::OutOfBounds(String::from("empty input")));
    }
    if x.len() != direction.len() {
        return Err(Error::DimensionMismatch {
            expected: x.len(),
            got: direction.len(),
        });
    }
    if x.iter().any(|value| !value.is_finite()) || direction.iter().any(|value| !value.is_finite())
    {
        return Err(Error::OutOfBounds(String::from("non-finite input")));
    }
    let seeded: Vec<Dual64> = x
        .iter()
        .zip(direction.iter())
        .map(|(&re, &eps)| Dual64::new(re, eps))
        .collect();
    let image = map(&seeded);
    let mut value = Vec::with_capacity(image.len());
    let mut derivative = Vec::with_capacity(image.len());
    for component in image {
        if !component.re.is_finite() || !component.eps.is_finite() {
            return Err(Error::OutOfBounds(String::from("non-finite output")));
        }
        value.push(component.re);
        derivative.push(component.eps);
    }
    Ok(DirectionalDerivative { value, derivative })
}

/// A scalar field `R^n -> R` written generically over dual numbers.
pub trait DualField: Send + Sync {
    /// Evaluates `f` at `x`. Instantiated at `f64` for values and at
    /// [`num_dual::DualDVec64`] for an exact gradient in one sweep.
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

    /// Sum over `components` of `ops · ε · scale`.
    fn summed_bar(ops: u64, scale: f64, components: usize) -> f64 {
        ops as f64 * f64::EPSILON * scale * components as f64
    }

    fn sup_abs(values: &[f64]) -> f64 {
        let mut scale = 0.0;
        for value in values {
            let magnitude = value.abs();
            if magnitude > scale {
                scale = magnitude;
            }
        }
        scale
    }

    #[test]
    fn scalar_sqrt_matches_the_analytic_derivative() {
        let jet = Dual64::new(4.0, 1.0).sqrt();
        let value_gap = (jet.re - 2.0).abs();
        let slope_gap = (jet.eps - 0.25).abs();
        let gap = value_gap + slope_gap;
        // sqrt, recip, two multiplies by one half, then the chain-rule multiply.
        let bar = summed_bar(5, sup_abs(&[jet.re, jet.eps, 2.0, 0.25]), 2);
        assert!(gap <= bar);
    }

    #[test]
    fn directional_derivative_matches_the_jacobian_product() {
        let x = [4.0, 2.0];
        let direction = [0.5, -0.25];
        let result = directional_derivative(&x, &direction, |z| {
            let x = z[0];
            let y = z[1];
            vec![x * y, x / y, x.sqrt()]
        });
        assert!(result.is_ok());
        if let Ok(jet) = result {
            let value = [8.0, 2.0, 2.0];
            let derivative = [0.0, 0.5, 0.125];
            let mut gap = 0.0;
            for (got, truth) in jet.value.iter().zip(&value) {
                gap += (got - truth).abs();
            }
            for (got, truth) in jet.derivative.iter().zip(&derivative) {
                gap += (got - truth).abs();
            }
            let scale = sup_abs(&jet.value) + sup_abs(&jet.derivative);
            // One product, one quotient (recip and two multiplies), one square root.
            let bar = summed_bar(6, scale, value.len() + derivative.len());
            assert_eq!(jet.value.len(), value.len());
            assert_eq!(jet.derivative.len(), derivative.len());
            assert!(gap <= bar);
        }
    }

    #[test]
    fn directional_derivative_rejects_a_short_direction() {
        let result = directional_derivative(&[0.0, 1.0], &[0.0], |z| z.to_vec());
        assert!(matches!(
            result,
            Err(Error::DimensionMismatch {
                expected: 2,
                got: 1
            })
        ));
    }

    fn is_out_of_bounds(result: Result<DirectionalDerivative, Error>) -> bool {
        matches!(result, Err(Error::OutOfBounds(_)))
    }

    #[test]
    fn directional_derivative_rejects_an_empty_point() {
        let result = directional_derivative(&[], &[], |_| Vec::new());
        assert!(is_out_of_bounds(result));
    }

    #[test]
    fn directional_derivative_rejects_a_non_finite_point() {
        let result = directional_derivative(&[f64::NAN, 1.0], &[0.0, 1.0], |z| z.to_vec());
        assert!(is_out_of_bounds(result));
    }

    #[test]
    fn directional_derivative_rejects_a_non_finite_direction() {
        let result = directional_derivative(&[1.0], &[f64::INFINITY], |z| z.to_vec());
        assert!(is_out_of_bounds(result));
    }

    #[test]
    fn directional_derivative_rejects_a_square_root_at_zero() {
        let result = directional_derivative(&[0.0], &[1.0], |z| vec![z[0].sqrt()]);
        assert!(is_out_of_bounds(result));
    }

    #[test]
    fn directional_derivative_rejects_division_by_zero() {
        let result = directional_derivative(&[1.0, 0.0], &[1.0, 0.0], |z| vec![z[0] / z[1]]);
        assert!(is_out_of_bounds(result));
    }

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
