//! Position-value pairs: a typed `(Array1<T>, T)` tuple with helpers.

use ndarray::Array1;
use num_traits::Float;

/// A position in N-dimensional space paired with its objective-function value.
///
/// Used throughout `eindir-core` and downstream `anneal-core` to track the
/// current, candidate, and best-seen states of a sampler.
#[derive(Clone, Debug)]
pub struct FPair<T: Float> {
    /// The N-dimensional position.
    pub pos: Array1<T>,
    /// The objective-function value evaluated at `pos`.
    pub val: T,
}

impl<T: Float> FPair<T> {
    /// Constructs a new `FPair` from a position and value.
    pub fn new(pos: Array1<T>, val: T) -> Self {
        Self { pos, val }
    }
}
