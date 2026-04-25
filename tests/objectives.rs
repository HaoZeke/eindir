use eindir_core::{Objective, StybTang2D, Rastrigin, Rosenbrock, Ackley};
use ndarray::Array1;

const TOL: f64 = 1e-3;

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
    assert_eq!(s.dim(), s.bounds().dims);

    let r = Rastrigin::<7>::new();
    assert_eq!(r.dim(), r.bounds().dims);

    let rb = Rosenbrock::<3>::new();
    assert_eq!(rb.dim(), rb.bounds().dims);

    let a = Ackley::<2>::new();
    assert_eq!(a.dim(), a.bounds().dims);
}
