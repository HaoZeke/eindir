use thiserror::Error;

/// Error type returned by all fallible operations in `eindir-core`.
#[derive(Debug, Error)]
pub enum Error {
    /// Returned when an input array's shape disagrees with the expected dimensionality.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Expected dimensionality, in number of elements along the first axis.
        expected: usize,
        /// Observed dimensionality.
        got: usize,
    },
    /// Returned when an index, parameter, or value lies outside its valid range.
    #[error("out of bounds: {0}")]
    OutOfBounds(String),
}
