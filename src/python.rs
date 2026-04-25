//! pyo3 entry point for `eindir._core`. Re-exports the package version and
//! typed bindings (FPair, Bounds, PyObjective) introduced in v0.3.0.

use pyo3::prelude::*;

use crate::py_objective::{PyBounds, PyFPair, PyObjective};

/// pyo3 module initialiser. Exposed to Python as `eindir._core`.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyBounds>()?;
    m.add_class::<PyFPair>()?;
    m.add_class::<PyObjective>()?;
    Ok(())
}
