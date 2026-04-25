//! N-dimensional Ackley: nearly flat outer region with a deep central well.

use ndarray::{Array1, ArrayView1};
use std::sync::OnceLock;
use crate::{Bounds, FPair, Objective};

/// N-dimensional Ackley: a multi-modal benchmark with a sharp central well.
/// Domain `[-32.768, 32.768]^D`. Global minimum at the origin with value `0`.
pub struct Ackley<const D: usize> {
    bounds: Bounds<f64>,
    min: OnceLock<FPair<f64>>,
}

impl<const D: usize> Ackley<D> {
    const A: f64 = 20.0;
    const B: f64 = 0.2;

    /// Constructs an Ackley objective in `D` dimensions.
    pub fn new() -> Self {
        Self {
            bounds: Bounds::new(
                Array1::from_vec(vec![-32.768; D]),
                Array1::from_vec(vec![ 32.768; D]),
                1e-9,
            ),
            min: OnceLock::new(),
        }
    }
}

impl<const D: usize> Default for Ackley<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize> Objective<f64> for Ackley<D> {
    fn dim(&self) -> usize {
        D
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let two_pi = 2.0 * std::f64::consts::PI;
        let n = D as f64;
        let sum_sq: f64 = x.iter().map(|xi| xi.powi(2)).sum();
        let sum_cos: f64 = x.iter().map(|xi| (two_pi * xi).cos()).sum();
        -Self::A * (-Self::B * (sum_sq / n).sqrt()).exp()
            - (sum_cos / n).exp()
            + Self::A
            + std::f64::consts::E
    }

    fn global_min(&self) -> Option<&FPair<f64>> {
        Some(self.min.get_or_init(|| FPair {
            pos: Array1::zeros(D),
            val: 0.0,
        }))
    }
}
