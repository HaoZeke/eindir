//! Three ways to give an objective a gradient: implement `Gradient` by hand
//! with a fused value and gradient, pair a closure with `AnalyticGradient`,
//! or difference the objective with `FiniteDiffGradient`. The gradients
//! how-to includes this file; `cargo run --example gradients`.
use eindir_core::{
    AnalyticGradient, Bounds, DifferentiableObjective, FiniteDiffGradient, Gradient, Objective,
};
use ndarray::{Array1, ArrayView1, array};

/// `f(x) = sum_i exp(x_i)`, whose gradient reuses the exponentials.
struct ExpSum {
    bounds: Bounds<f64>,
}

impl Objective<f64> for ExpSum {
    fn dim(&self) -> usize {
        self.bounds.dims
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        x.mapv(f64::exp).sum()
    }
}

impl Gradient<f64> for ExpSum {
    fn dim(&self) -> usize {
        self.bounds.dims
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        x.mapv(f64::exp)
    }
}

// No blanket impl exists, so opt in; override to fuse the two passes.
impl DifferentiableObjective<f64> for ExpSum {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let e = x.mapv(f64::exp);
        (e.sum(), e)
    }
}

fn main() {
    let bounds = Bounds::new(array![-1.0, -1.0], array![1.0, 1.0], 0.0);
    let f = ExpSum {
        bounds: bounds.clone(),
    };
    let x = array![0.0, 0.5];

    // Objective::dim, since f.dim() would be ambiguous here.
    assert_eq!(Objective::dim(&f), 2);
    let (value, g) = f.value_and_gradient(x.view());
    println!("f = {value:.6}, grad = {g}");

    let closure = AnalyticGradient::new(2, |x: ArrayView1<f64>| x.mapv(f64::exp));
    assert_eq!(closure.grad(x.view()), g);

    // Central differences: 2 * dim evaluations, error of order h^2.
    let fd = FiniteDiffGradient::new(ExpSum { bounds });
    let err = (&fd.grad(x.view()) - &g).mapv(f64::abs).sum();
    println!("finite-difference error {err:.1e}");
    assert!(err < 1e-8);
}
