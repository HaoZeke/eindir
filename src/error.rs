use thiserror::Error;

/// Shape and range errors for crates built on `eindir-core`.
///
/// No function in `eindir-core` returns it: the constructors here panic on
/// bad input (see each `# Panics` section), and fallible objectives use their
/// own error types through [`crate::TryObjective`]. It is exported for
/// downstream crates that want a shared error vocabulary.
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
