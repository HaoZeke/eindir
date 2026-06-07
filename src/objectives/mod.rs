//! Builtin objective-function implementations for benchmarking and tests.

pub mod ackley;
pub mod rastrigin;
pub mod rosenbrock;
pub mod styblinski_tang;

pub use ackley::Ackley;
pub use rastrigin::Rastrigin;
pub use rosenbrock::Rosenbrock;
pub use styblinski_tang::StybTang2D;
