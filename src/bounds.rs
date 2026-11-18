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
    /// Constructs the box `[low, high]` with membership tolerance `slack`.
    ///
    /// `slack` is in the same units as the coordinates. It widens
    /// [`Bounds::contains`] only; [`Bounds::clip`] and [`Bounds::mkpoint`]
    /// use the box without it. `low[i] == high[i]` is accepted.
    ///
    /// # Panics
    ///
    /// Panics when `low` and `high` differ in length, when `slack` is
    /// negative, and when `low[i]` exceeds `high[i]` for any `i`.
    ///
    /// # Examples
    ///
    /// ```
    /// use eindir_core::Bounds;
    /// use ndarray::array;
    ///
    /// let b = Bounds::new(array![-1.0, 0.0], array![1.0, 2.0], 1e-6);
    /// assert_eq!(b.dims, 2);
    /// assert!(b.contains(array![1.0 + 5e-7, 2.0].view()));
    /// assert_eq!(b.clip(array![3.0, -1.0].view()), array![1.0, 0.0]);
    /// ```
    pub fn new(low: Array1<T>, high: Array1<T>, slack: T) -> Self {
        let dims = low.len();
        assert_eq!(low.len(), high.len(), "low and high have different shapes");
        assert!(slack >= T::zero(), "slack must be non-negative");
        if let Some(index) = low.iter().zip(high.iter()).position(|(&lo, &hi)| lo > hi) {
            panic!(
                "low[{index}] must be at most high[{index}]: low must not exceed high in any coordinate"
            );
        }
        Self {
            low,
            high,
            slack,
            dims,
        }
    }

    /// True if `x` has `dims` coordinates and each lies in
    /// `[low[i] - slack, high[i] + slack]`.
    ///
    /// A point of the wrong length gives `false`, not a panic.
    pub fn contains(&self, x: ArrayView1<T>) -> bool {
        x.len() == self.dims
            && x.iter()
                .enumerate()
                .all(|(i, &xi)| xi >= self.low[i] - self.slack && xi <= self.high[i] + self.slack)
    }

    /// Draws a point uniformly from the half-open box `[low, high)`.
    ///
    /// # Panics
    ///
    /// Panics when `low[i] == high[i]` for any `i`, because the uniform
    /// distribution on an empty interval is undefined.
    pub fn mkpoint<R: Rng>(&self, rng: &mut R) -> Array1<T> {
        Array1::from_iter((0..self.dims).map(|i| {
            Uniform::new(self.low[i], self.high[i])
                .expect("low < high required for Uniform sampling")
                .sample(rng)
        }))
    }

    /// Clips each coordinate of `x` to the closed `[low, high]` interval.
    ///
    /// `slack` plays no part. A NaN coordinate becomes `low[i]`, following
    /// [`Float::max`].
    ///
    /// # Panics
    ///
    /// Panics when `x.len() != dims`.
    ///
    /// # Examples
    ///
    /// ```
    /// use eindir_core::Bounds;
    /// use ndarray::array;
    ///
    /// let b = Bounds::new(array![0.0, 0.0], array![1.0, 1.0], 0.0);
    /// assert_eq!(b.clip(array![f64::NAN, 0.5].view()), array![0.0, 0.5]);
    /// ```
    pub fn clip(&self, x: ArrayView1<T>) -> Array1<T> {
        assert_eq!(x.len(), self.dims, "point has the wrong dimension");
        Array1::from_iter(
            x.iter()
                .enumerate()
                .map(|(i, &xi)| xi.max(self.low[i]).min(self.high[i])),
        )
    }
}
