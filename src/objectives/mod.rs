//! Builtin objective-function implementations for benchmarking and tests.

pub mod styblinski_tang;
pub mod rastrigin;
pub mod rosenbrock;
pub mod ackley;

pub use styblinski_tang::StybTang2D;
pub use rastrigin::Rastrigin;
pub use rosenbrock::Rosenbrock;
pub use ackley::Ackley;
