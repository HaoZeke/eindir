//! Styblinski-Tang 2D: a non-convex benchmark with a single global minimum.

use ndarray::{Array1, ArrayView1};
use std::sync::OnceLock;
use crate::{Bounds, FPair, Objective};

/// 2D Styblinski-Tang objective `(1/2) * sum_i (x_i^4 - 16 x_i^2 + 5 x_i)`.
///
/// Domain `[-5, 5]^2`. Global minimum at `x = (-2.903534, -2.903534)` with
/// value `-39.16599 * 2 = -78.33198`.
pub struct StybTang2D {
    bounds: Bounds<f64>,
    min: OnceLock<FPair<f64>>,
}

impl StybTang2D {
    /// Constructs the canonical 2D Styblinski-Tang objective.
    pub fn new() -> Self {
        Self {
            bounds: Bounds::new(
                Array1::from_vec(vec![-5.0, -5.0]),
                Array1::from_vec(vec![ 5.0,  5.0]),
                1e-9,
            ),
            min: OnceLock::new(),
        }
    }
}

impl Default for StybTang2D {
    fn default() -> Self {
        Self::new()
    }
}

impl Objective<f64> for StybTang2D {
    fn dim(&self) -> usize {
        2
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        x.iter()
            .map(|xi| xi.powi(4) - 16.0 * xi.powi(2) + 5.0 * xi)
            .sum::<f64>()
            / 2.0
    }

    fn global_min(&self) -> Option<&FPair<f64>> {
        Some(self.min.get_or_init(|| FPair {
            pos: Array1::from_vec(vec![-2.903534, -2.903534]),
            val: -39.16599 * 2.0,
        }))
    }
}
