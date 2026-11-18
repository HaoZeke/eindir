//! The coloured-noise GLE thermostat on its own: build the optimal-sampling
//! drift, start from the stationary distribution and check that the
//! physical momentum keeps its thermal variance. The GLE tutorial includes
//! this file; `cargo run --example gle_thermostat`.
use eindir_core::{GleThermostat, OPTIMAL_SAMPLING_NS, optimal_sampling_drift};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    // Flat sampling efficiency over [omega0, 100 omega0], in inverse time units.
    let omega0 = 0.2;
    let drift = optimal_sampling_drift(omega0);
    assert_eq!(drift.nrows(), OPTIMAL_SAMPLING_NS + 1);

    // Reduced units: kb = 1, so the momentum variance is the temperature.
    let (dt, temperature, kb) = (0.01, 1.0, 1.0);
    let gle = GleThermostat::canonical(&drift, dt, temperature, kb);

    // Row 0 is the mass-scaled physical momentum, rows 1..=ns the auxiliary ones.
    let covariance = Array2::<f64>::eye(gle.ns + 1) * temperature;
    let dim = 4;
    let mut state = gle.sample_stationary(&covariance, dim, kb, &mut rng);

    let steps = 20_000;
    let mut p2 = 0.0;
    for _ in 0..steps {
        gle.step(&mut state.view_mut(), &mut rng);
        p2 += state.row(0).mapv(|p| p * p).sum();
    }
    let mean_p2 = p2 / (steps * dim) as f64;
    println!("<p^2> = {mean_p2:.3} over {steps} steps, kB T = {temperature}");
    assert!((mean_p2 - temperature).abs() < 0.1);
}
