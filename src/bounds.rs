//! Box bounds on N-dimensional points with sampling and clipping.

use ndarray::{Array1, ArrayView1};
use num_traits::Float;
use rand::Rng;
use rand_distr::uniform::SampleUniform;
use rand_distr::{Distribution, Uniform};

/// Why a [`Bounds`] cannot be built from the arrays it was given.
///
/// Further faults may be added, so callers match with a final arm.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum BoundsError {
    /// `low` and `high` have different lengths.
    #[error("low and high have different shapes ({low} and {high})")]
    ShapeMismatch {
        /// Length of `low`.
        low: usize,
        /// Length of `high`.
        high: usize,
    },
    /// `slack` is negative or not a number.
    #[error("slack must be non-negative")]
    NegativeSlack,
    /// A coordinate of `low` or `high` is not a number.
    #[error("low[{index}] and high[{index}] must be numbers")]
    NotANumber {
        /// Coordinate at which the first NaN sits.
        index: usize,
    },
    /// A coordinate has `low[index] > high[index]`.
    #[error("low[{index}] exceeds high[{index}]: low must not exceed high in any coordinate")]
    Inverted {
        /// Coordinate of the first inverted pair.
        index: usize,
    },
}

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
    /// Constructs `Bounds`, or says why the box cannot exist.
    ///
    /// The sides must have one length, every coordinate must be a number
    /// with `low[i] <= high[i]`, and `slack` must be a number that is not
    /// negative. Equal sides are a box of zero width in that coordinate.
    ///
    /// Infinity is not a NaN. A positive infinite slack is accepted, and
    /// a negative infinite slack is [`BoundsError::NegativeSlack`]. An
    /// infinite coordinate is accepted when `low[i] <= high[i]` in IEEE
    /// order, so a side may run to `+∞`, start at `-∞`, be the whole
    /// line, or be a single infinity. `+∞` above a finite value or above
    /// `-∞` is [`BoundsError::Inverted`].
    ///
    /// ```
    /// use eindir_core::{Bounds, BoundsError};
    /// use ndarray::array;
    ///
    /// let ok = Bounds::try_new(array![0.0, 0.0], array![1.0, 2.0], 0.0).unwrap();
    /// assert_eq!(ok.dims, 2);
    /// let err = Bounds::try_new(array![1.0], array![0.0], 0.0).unwrap_err();
    /// assert_eq!(err, BoundsError::Inverted { index: 0 });
    /// ```
    pub fn try_new(low: Array1<T>, high: Array1<T>, slack: T) -> Result<Self, BoundsError> {
        let dims = low.len();
        if low.len() != high.len() {
            return Err(BoundsError::ShapeMismatch {
                low: low.len(),
                high: high.len(),
            });
        }
        if slack.is_nan() || slack < T::zero() {
            return Err(BoundsError::NegativeSlack);
        }
        if let Some(index) = low
            .iter()
            .zip(high.iter())
            .position(|(&lo, &hi)| lo.is_nan() || hi.is_nan())
        {
            return Err(BoundsError::NotANumber { index });
        }
        if let Some(index) = low.iter().zip(high.iter()).position(|(&lo, &hi)| lo > hi) {
            return Err(BoundsError::Inverted { index });
        }
        Ok(Self {
            low,
            high,
            slack,
            dims,
        })
    }

    /// Constructs `Bounds` and panics with the message of the
    /// [`BoundsError`] that [`Bounds::try_new`] returns for the same input.
    pub fn new(low: Array1<T>, high: Array1<T>, slack: T) -> Self {
        Self::try_new(low, high, slack).unwrap_or_else(|err| panic!("{err}"))
    }

    /// True if every coordinate of `x` lies within `slack` of the box.
    ///
    /// Membership subtracts `slack` from `low` and adds it to `high`.
    /// A positive infinite slack on a finite face makes that side
    /// unbounded, so every non-NaN coordinate passes. `+∞ - +∞` is NaN,
    /// and a comparison with NaN is false, so a positive infinite slack
    /// on a `+∞` face matches no point.
    pub fn contains(&self, x: ArrayView1<T>) -> bool {
        x.len() == self.dims
            && x.iter()
                .enumerate()
                .all(|(i, &xi)| xi >= self.low[i] - self.slack && xi <= self.high[i] + self.slack)
    }

    /// Draws a uniform random point inside the box.
    ///
    /// Panics when [`Uniform::new`] rejects an endpoint. It rejects a
    /// zero-width coordinate and a non-finite one, so a box that
    /// [`Bounds::try_new`] accepted with an infinite side cannot be
    /// sampled. Slack is not used.
    pub fn mkpoint<R: Rng>(&self, rng: &mut R) -> Array1<T> {
        Array1::from_iter((0..self.dims).map(|i| {
            Uniform::new(self.low[i], self.high[i])
                .expect("low < high required for Uniform sampling")
                .sample(rng)
        }))
    }

    /// Clips each coordinate of `x` to the closed `[low, high]` interval.
    ///
    /// Slack is not used. A finite value on an infinite side is left as
    /// it is, and a value past a finite face becomes that face.
    pub fn clip(&self, x: ArrayView1<T>) -> Array1<T> {
        assert_eq!(x.len(), self.dims, "point has the wrong dimension");
        Array1::from_iter(
            x.iter()
                .enumerate()
                .map(|(i, &xi)| xi.max(self.low[i]).min(self.high[i])),
        )
    }
}
