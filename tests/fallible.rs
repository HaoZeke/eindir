//! The fallible path as a downstream crate sees it: an own error type, a
//! driver that returns it unchanged, and the blanket impls for infallible
//! types.

use eindir_core::{
    Bounds, DifferentiableObjective, Gradient, NanOnError, Objective, Rosenbrock,
    TryDifferentiableObjective, TryFiniteDiffGradient, TryGradient, TryObjective,
};
use ndarray::{Array1, ArrayView1, array};
use std::convert::Infallible;

const TOLERANCE: f64 = 1.0 / 1_000_000.0;

/// A caller's error, with data that a box or a string would lose.
#[derive(Debug, Clone, PartialEq)]
enum PesError {
    Refused { atom: usize, distance: f64 },
    Engine(i32),
}

struct Pes {
    bounds: Bounds<f64>,
}

impl Pes {
    fn new() -> Self {
        Self {
            bounds: Bounds::new(array![-5.0, -5.0], array![5.0, 5.0], 0.0),
        }
    }

    fn check(&self, x: ArrayView1<f64>) -> Result<(), PesError> {
        if x.iter().any(|v| !v.is_finite()) {
            return Err(PesError::Engine(-1));
        }
        if x[0] < -4.0 {
            return Err(PesError::Refused {
                atom: 0,
                distance: -x[0],
            });
        }
        Ok(())
    }
}

impl TryObjective<f64> for Pes {
    type Error = PesError;

    fn dims(&self) -> usize {
        2
    }

    fn domain(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn try_eval(&self, x: ArrayView1<f64>) -> Result<f64, PesError> {
        self.check(x)?;
        Ok(x.dot(&x))
    }
}

impl TryGradient<f64> for Pes {
    type GradError = PesError;

    fn try_grad(&self, x: ArrayView1<f64>) -> Result<Array1<f64>, PesError> {
        self.check(x)?;
        Ok(x.mapv(|v| 2.0 * v))
    }

    fn grad_dims(&self) -> usize {
        2
    }
}

impl TryDifferentiableObjective<f64> for Pes {}

/// The shape of a solver step: generic over the objective, returning the
/// objective's own error type.
fn descent_step<O>(obj: &O, x: &mut Array1<f64>, rate: f64) -> Result<f64, O::Error>
where
    O: TryDifferentiableObjective<f64> + ?Sized,
{
    let (value, grad) = obj.try_value_and_gradient(x.view())?;
    *x -= &(grad * rate);
    Ok(value)
}

#[test]
fn a_custom_error_reaches_the_caller_unchanged() {
    let pes = Pes::new();
    let mut x = array![1.0, 1.0];
    assert_eq!(descent_step(&pes, &mut x, 0.25), Ok(2.0));
    assert_eq!(x, array![0.5, 0.5]);

    let mut refused = array![-4.5, 0.0];
    assert_eq!(
        descent_step(&pes, &mut refused, 0.25),
        Err(PesError::Refused {
            atom: 0,
            distance: 4.5
        })
    );
    assert_eq!(refused, array![-4.5, 0.0], "a failed step leaves x alone");

    let mut broken = array![f64::NAN, 0.0];
    assert_eq!(
        descent_step(&pes, &mut broken, 0.25),
        Err(PesError::Engine(-1))
    );
}

#[test]
fn the_driver_works_through_a_trait_object() {
    let pes = Pes::new();
    let dynamic: &dyn TryDifferentiableObjective<f64, Error = PesError, GradError = PesError> =
        &pes;
    let mut x = array![2.0, 0.0];
    assert_eq!(descent_step(dynamic, &mut x, 0.5), Ok(4.0));
}

struct Bowl {
    bounds: Bounds<f64>,
}

impl Objective<f64> for Bowl {
    fn dim(&self) -> usize {
        2
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        x.dot(&x)
    }
}

impl Gradient<f64> for Bowl {
    fn dim(&self) -> usize {
        2
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        x.mapv(|v| 2.0 * v)
    }
}

impl DifferentiableObjective<f64> for Bowl {}

#[test]
fn an_infallible_type_is_accepted_by_the_same_driver() {
    let bowl = Bowl {
        bounds: Bounds::new(array![-5.0, -5.0], array![5.0, 5.0], 0.0),
    };
    let mut x = array![1.0, 1.0];
    let outcome: Result<f64, Infallible> = descent_step(&bowl, &mut x, 0.25);
    assert_eq!(outcome, Ok(2.0));
    assert_eq!(x, array![0.5, 0.5]);
}

#[test]
fn a_builtin_objective_is_a_try_objective() {
    let rosenbrock = Rosenbrock::<2>::new();
    let outcome: Result<f64, Infallible> = rosenbrock.try_eval(array![1.0, 1.0].view());
    assert_eq!(outcome, Ok(0.0));
}

#[test]
fn the_infallible_adapters_round_trip() {
    let adapted = NanOnError::new(Pes::new(), |_error: PesError| {});
    assert_eq!(adapted.eval(array![1.0, 2.0].view()), 5.0);
    assert!(adapted.eval(array![-4.5, 0.0].view()).is_nan());
    let back: Result<f64, Infallible> = adapted.try_eval(array![1.0, 2.0].view());
    assert_eq!(back, Ok(5.0));
}

#[test]
fn central_differences_over_a_fallible_objective_stop_at_the_error() {
    let fd = TryFiniteDiffGradient::new(Pes::new());
    let g = fd
        .try_grad(array![1.0, 2.0].view())
        .expect("interior point");
    assert!((g[0] - 2.0).abs() < TOLERANCE && (g[1] - 4.0).abs() < TOLERANCE);
    assert_eq!(
        fd.try_grad(array![-4.0, 0.0].view()),
        Err(PesError::Refused {
            atom: 0,
            distance: 4.0 + fd.step()
        })
    );
}
