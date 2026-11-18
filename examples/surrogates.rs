//! An additive Chebyshev surrogate fitted on a Halton pilot, its tempered
//! independence proposals, and a one-dimensional reduced objective. The
//! surrogates tutorial includes this file; `cargo run --example surrogates`.
use eindir_core::{
    AdditiveSurrogate, Bounds, Objective, Rastrigin, ReducedObjective, low_discrepancy_points,
};
use ndarray::{Array1, ArrayView1, array};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() {
    let bounds = Bounds::new(array![-2.0, -2.0, -2.0], array![2.0, 2.0, 2.0], 0.0);
    let f = |x: ArrayView1<f64>| x[0] * x[0] + (2.0 * x[1]).sin() + 0.5 * x[2];

    // Fit c + sum_j g_j(x_j), each g_j a degree-8 Chebyshev series.
    let pilot = low_discrepancy_points(&bounds, 128, 1);
    let y: Array1<f64> = pilot.outer_iter().map(f).collect();
    let surrogate = AdditiveSurrogate::fit(pilot.view(), y.view(), bounds.clone(), 8);

    // The function is separable, so the fit is close on points it has not seen.
    let fresh = low_discrepancy_points(&bounds, 64, 1000);
    let worst = fresh
        .outer_iter()
        .map(|x| (surrogate.value(x) - f(x)).abs())
        .fold(0.0, f64::max);
    println!("worst error {worst:.2e} on 64 new points");
    assert!(worst < 1e-2);

    // Independence proposals from exp(-(f - f_min) / T), one coordinate at a time.
    let mut rng = StdRng::seed_from_u64(0);
    let proposals = surrogate.sample(5, 1.0, 256, &mut rng);
    assert!(proposals.outer_iter().all(|x| bounds.contains(x)));

    // Search 2-D Rastrigin along the diagonal x = (r, r) only.
    let reduced = ReducedObjective::new(
        Rastrigin::<2>::new(),
        array![0.0, 0.0],
        array![[1.0], [1.0]],
        Bounds::new(array![-5.12], array![5.12], 0.0),
    );
    assert_eq!(Objective::dim(&reduced), 1);
    println!(
        "reduced Rastrigin at r = 1: {}",
        reduced.eval(array![1.0].view())
    );
}
