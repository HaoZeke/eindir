//! The GLE coloured-noise thermostat and the additive surrogate, which
//! have no Python binding.

use eindir_core::{
    AdditiveSurrogate, Bounds, GleThermostat, OPTIMAL_SAMPLING_NS, Objective,
    low_discrepancy_points, optimal_sampling_drift,
};
use ndarray::{Array1, Array2, ArrayView1, array};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    let omega0 = 0.2;
    let drift = optimal_sampling_drift(omega0);
    let n = OPTIMAL_SAMPLING_NS + 1;
    assert_eq!(drift.dim(), (n, n));

    let (dt, temperature, kb) = (0.01, 1.0, 1.0);
    let gle = GleThermostat::canonical(&drift, dt, temperature, kb);
    let covariance = Array2::<f64>::eye(gle.ns + 1) * temperature;
    let dim = 4;
    let mut state = gle.sample_stationary(&covariance, dim, kb, &mut rng);
    assert_eq!(state.dim(), (gle.ns + 1, dim));
    let steps = 20_000;
    let mut p2 = 0.0;
    for _ in 0..steps {
        gle.step(&mut state.view_mut(), &mut rng);
        p2 += state.row(0).mapv(|p| p * p).sum();
    }
    let mean_p2 = p2 / (steps * dim) as f64;
    println!("GLE: <p^2> = {mean_p2:.3} over {steps} steps, kB T = {temperature}");

    let bounds = Bounds::new(array![-2.0, -2.0, -2.0], array![2.0, 2.0, 2.0], 0.0);
    let f = |x: ArrayView1<f64>| x[0] * x[0] + (2.0 * x[1]).sin() + 0.5 * x[2];
    let pilot = low_discrepancy_points(&bounds, 128, 1);
    let y: Array1<f64> = pilot.outer_iter().map(f).collect();
    let degree = 8;
    let surrogate = AdditiveSurrogate::fit(pilot.view(), y.view(), bounds.clone(), degree);
    let fresh = low_discrepancy_points(&bounds, 64, 1000);
    let worst = fresh
        .outer_iter()
        .map(|x| (surrogate.eval(x) - f(x)).abs())
        .fold(0.0, f64::max);
    println!("additive surrogate, degree {degree}: worst error {worst:.2e} on 64 new points");

    let proposals = surrogate.sample(5, 1.0, 256, &mut rng);
    assert_eq!(proposals.dim(), (5, 3));
    assert!(proposals.outer_iter().all(|x| bounds.contains(x)));
}
