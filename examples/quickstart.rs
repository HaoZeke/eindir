//! A sphere objective on a 2-D box, evaluated at eight Halton start points.
//! The manual's Quickstart includes this file; `cargo run --example quickstart`.
use eindir_core::{Bounds, Objective, low_discrepancy_points};
use ndarray::{ArrayView1, array};

struct Sphere {
    bounds: Bounds<f64>,
}

impl Objective<f64> for Sphere {
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        x.iter().map(|xi| (xi - 1.0).powi(2)).sum()
    }
}

fn main() {
    let bounds = Bounds::new(array![-5.12, -5.12], array![5.12, 5.12], 1e-6);
    let obj = Sphere { bounds };
    let starts = low_discrepancy_points(obj.bounds(), 8, 1);
    let values = obj.eval_batch(starts.view());
    let best = (0..values.len())
        .min_by(|&a, &b| values[a].total_cmp(&values[b]))
        .unwrap();
    println!("{} {}", starts.row(best), values[best]);
}
