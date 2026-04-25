//! N-dimensional Rosenbrock: classic banana-shaped valley benchmark.

use ndarray::{Array1, ArrayView1};
use std::sync::OnceLock;
use crate::{Bounds, FPair, Objective};

/// N-dimensional Rosenbrock: `sum_{i=0}^{D-2} (100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2)`.
/// Domain `[-2.048, 2.048]^D`. Global minimum at `x = (1, 1, ..., 1)` with value `0`.
pub struct Rosenbrock<const D: usize> {
    bounds: Bounds<f64>,
    min: OnceLock<FPair<f64>>,
}

impl<const D: usize> Rosenbrock<D> {
    /// Constructs a Rosenbrock objective in `D` dimensions.
    pub fn new() -> Self {
        Self {
            bounds: Bounds::new(
                Array1::from_vec(vec![-2.048; D]),
                Array1::from_vec(vec![ 2.048; D]),
                1e-9,
            ),
            min: OnceLock::new(),
        }
    }
}

impl<const D: usize> Default for Rosenbrock<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize> Objective<f64> for Rosenbrock<D> {
    fn dim(&self) -> usize {
        D
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        (0..D - 1)
            .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
            .sum()
    }

    fn global_min(&self) -> Option<&FPair<f64>> {
        Some(self.min.get_or_init(|| FPair {
            pos: Array1::ones(D),
            val: 0.0,
        }))
    }
}
