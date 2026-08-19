use eindir_core::{Ackley, Gradient, Objective, Rastrigin, Rosenbrock, StybTang2D};
use ndarray::Array1;

const TOL: f64 = 1e-3;

fn finite_difference<O: Objective<f64>>(obj: &O, x: &Array1<f64>) -> Array1<f64> {
    let h = 1e-6;
    let mut grad = Array1::zeros(x.len());
    for i in 0..x.len() {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[i] += h;
        xm[i] -= h;
        grad[i] = (obj.eval(xp.view()) - obj.eval(xm.view())) / (2.0 * h);
    }
    grad
}

fn assert_gradient_matches_finite_difference<O>(obj: &O, x: Array1<f64>, tol: f64)
where
    O: Objective<f64> + Gradient<f64>,
{
    let actual = obj.grad(x.view());
    let expected = finite_difference(obj, &x);
    for i in 0..x.len() {
        assert!(
            (actual[i] - expected[i]).abs() < tol,
            "dim {i}: analytic {} vs finite-difference {}",
            actual[i],
            expected[i]
        );
    }
}

#[test]
fn styb_tang_global_min() {
    let obj = StybTang2D::new();
    let min = obj.global_min().expect("StybTang2D has a known minimum");
    let v = obj.eval(min.pos.view());
    assert!((v - min.val).abs() < TOL, "got {}, expected {}", v, min.val);
}

#[test]
fn rastrigin_global_min_at_origin() {
    let obj = Rastrigin::<3>::new();
    let v = obj.eval(Array1::zeros(3).view());
    assert!(v.abs() < 1e-12);
}

#[test]
fn rosenbrock_global_min_at_ones() {
    let obj = Rosenbrock::<4>::new();
    let v = obj.eval(Array1::ones(4).view());
    assert!(v.abs() < 1e-12);
}

#[test]
fn ackley_global_min_at_origin() {
    let obj = Ackley::<5>::new();
    let v = obj.eval(Array1::zeros(5).view());
    assert!(v.abs() < 1e-9, "ackley(0) was {}", v);
}

#[test]
fn objectives_dim_matches_bounds() {
    let s = StybTang2D::new();
    assert_eq!(Objective::dim(&s), s.bounds().dims);

    let r = Rastrigin::<7>::new();
    assert_eq!(Objective::dim(&r), r.bounds().dims);

    let rb = Rosenbrock::<3>::new();
    assert_eq!(Objective::dim(&rb), rb.bounds().dims);

    let a = Ackley::<2>::new();
    assert_eq!(Objective::dim(&a), a.bounds().dims);
}

#[test]
fn builtin_gradients_match_finite_differences() {
    assert_gradient_matches_finite_difference(
        &StybTang2D::new(),
        Array1::from_vec(vec![-1.2, 0.7]),
        1e-5,
    );
    assert_gradient_matches_finite_difference(
        &Rastrigin::<3>::new(),
        Array1::from_vec(vec![0.2, -0.7, 1.1]),
        1e-5,
    );
    assert_gradient_matches_finite_difference(
        &Rosenbrock::<4>::new(),
        Array1::from_vec(vec![0.8, 1.2, -0.4, 0.3]),
        1e-4,
    );
    assert_gradient_matches_finite_difference(
        &Ackley::<4>::new(),
        Array1::from_vec(vec![0.2, -1.1, 0.7, 1.4]),
        1e-5,
    );
}

#[test]
fn rosenbrock_one_dimensional_gradient_is_zero() {
    let obj = Rosenbrock::<1>::new();
    let grad = obj.grad(Array1::from_vec(vec![0.25]).view());
    assert_eq!(grad.len(), 1);
    assert!(grad[0].abs() < 1e-12);
}

#[test]
#[should_panic(expected = "Ackley requires at least one dimension")]
fn ackley_rejects_zero_dimensions() {
    let _ = Ackley::<0>::new();
}

#[test]
#[should_panic(expected = "Rosenbrock requires at least one dimension")]
fn rosenbrock_rejects_zero_dimensions() {
    let _ = Rosenbrock::<0>::new();
}

#[test]
fn eval_batch_parallel_matches_serial_rastrigin() {
    use eindir_core::{Rastrigin, eval_batch_parallel};
    use ndarray::Array2;

    let obj = Rastrigin::<8>::new();
    let bounds = obj.bounds();
    let starts = eindir_core::low_discrepancy_points(bounds, 64, 1);
    let serial = obj.eval_batch(starts.view());
    let parallel = eval_batch_parallel(&obj, starts.view());
    assert_eq!(serial.len(), parallel.len());
    for (a, b) in serial.iter().zip(parallel.iter()) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }
    let tiny = Array2::<f64>::zeros((0, 8));
    assert_eq!(eval_batch_parallel(&obj, tiny.view()).len(), 0);
}
