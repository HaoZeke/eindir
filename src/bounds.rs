//! Box bounds on N-dimensional points with sampling and clipping.

use ndarray::{Array1, ArrayView1};
use num_traits::Float;
use rand::Rng;
use rand_distr::uniform::SampleUniform;
use rand_distr::{Distribution, Uniform};

/// Box bounds on an N-dimensional position.
///
/// `Bounds` represents the hyperrectangle `[low, high]` inflated by `slack`
/// for membership tests, supporting uniform sampling and clipping. The
/// dimensionality `dims` matches `low.len() == high.len()`.
#[derive(Clone, Debug)]
pub struct Bounds<T: Float> {
    /// Lower corner of the box.
    pub low: Array1<T>,
    /// Upper corner of the box.
    pub high: Array1<T>,
    /// Tolerance for membership tests; a point is considered in-bounds if
    /// each coordinate is within `slack` of the corresponding `low`/`high`.
    pub slack: T,
    /// Number of dimensions, equal to `low.len()`.
    pub dims: usize,
}

impl<T> Bounds<T>
where
    T: Float + SampleUniform + 'static,
    Uniform<T>: Distribution<T>,
{
    /// Constructs `Bounds` and asserts `low.len() == high.len()`.
    pub fn new(low: Array1<T>, high: Array1<T>, slack: T) -> Self {
        let dims = low.len();
        assert_eq!(low.len(), high.len(), "low and high have different shapes");
        Self { low, high, slack, dims }
    }

    /// True if every coordinate of `x` lies within `slack` of the box.
    pub fn contains(&self, x: ArrayView1<T>) -> bool {
        x.iter().enumerate().all(|(i, &xi)| {
            xi >= self.low[i] - self.slack && xi <= self.high[i] + self.slack
        })
    }

    /// Draws a uniform random point inside the box.
    pub fn mkpoint<R: Rng>(&self, rng: &mut R) -> Array1<T> {
        Array1::from_iter((0..self.dims).map(|i| {
            Uniform::new(self.low[i], self.high[i])
                .expect("low < high required for Uniform sampling")
                .sample(rng)
        }))
    }

    /// Clips each coordinate of `x` to the closed `[low, high]` interval.
    pub fn clip(&self, x: ArrayView1<T>) -> Array1<T> {
        Array1::from_iter(x.iter().enumerate().map(|(i, &xi)| {
            xi.max(self.low[i]).min(self.high[i])
        }))
    }
}
