//! pyo3 entry point for `eindir._core`. Re-exports the package version and
//! typed bindings for FPair, Bounds, PyObjective, and point sets.

use pyo3::prelude::*;

use crate::py_objective::{
    PyBounds, PyFPair, PyObjective, low_discrepancy_points, shifted_low_discrepancy_points,
};

/// The Rust extension of eindir, built from `eindir-core`.
///
/// It holds ``__version__``, ``Bounds``, ``FPair``, ``PyObjective``,
/// ``low_discrepancy_points`` and ``shifted_low_discrepancy_points``.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyBounds>()?;
    m.add_class::<PyFPair>()?;
    m.add_class::<PyObjective>()?;
    m.add_function(wrap_pyfunction!(low_discrepancy_points, m)?)?;
    m.add_function(wrap_pyfunction!(shifted_low_discrepancy_points, m)?)?;
    Ok(())
}
