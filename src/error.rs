use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("out of bounds: {0}")]
    OutOfBounds(String),
}
