//! N-dimensional Rastrigin: highly multimodal benchmark.

use ndarray::{Array1, ArrayView1};
use std::sync::OnceLock;
use crate::{Bounds, FPair, Objective};

/// N-dimensional Rastrigin: `A * D + sum_i (x_i^2 - A * cos(2 * pi * x_i))`
/// with `A = 10`. Domain `[-5.12, 5.12]^D`. Global minimum at the origin
/// with value `0`.
pub struct Rastrigin<const D: usize> {
    bounds: Bounds<f64>,
    min: OnceLock<FPair<f64>>,
}

impl<const D: usize> Rastrigin<D> {
    const A: f64 = 10.0;

    /// Constructs a Rastrigin objective in `D` dimensions.
    pub fn new() -> Self {
        Self {
            bounds: Bounds::new(
                Array1::from_vec(vec![-5.12; D]),
                Array1::from_vec(vec![ 5.12; D]),
                1e-9,
            ),
            min: OnceLock::new(),
        }
    }
}

impl<const D: usize> Default for Rastrigin<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize> Objective<f64> for Rastrigin<D> {
    fn dim(&self) -> usize {
        D
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let two_pi = 2.0 * std::f64::consts::PI;
        let inner: f64 = x
            .iter()
            .map(|xi| xi.powi(2) - Self::A * (two_pi * xi).cos())
            .sum();
        Self::A * (D as f64) + inner
    }

    fn global_min(&self) -> Option<&FPair<f64>> {
        Some(self.min.get_or_init(|| FPair {
            pos: Array1::zeros(D),
            val: 0.0,
        }))
    }
}
