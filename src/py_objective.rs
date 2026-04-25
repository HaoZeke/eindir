//! Adapter wrapping a Python callable into the Rust `Objective<f64>` trait.
//!
//! The numpy 0.26 crate is built against ndarray 0.16, while the rest of
//! eindir-core uses ndarray 0.17. To avoid an ABI mismatch between the two
//! ndarray copies, we cross the pyo3 boundary as `&[f64]` / `Vec<f64>` and
//! reconstruct ndarray 0.17 `Array1<f64>` / `ArrayView1<f64>` on the Rust
//! side via `Array1::from_vec` / `ArrayView1::from`. The dlpk-backed
//! zero-copy path can replace this in v0.4 once the workspace settles on a
//! single ndarray version end-to-end.

use ndarray::{Array1, ArrayView1};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use crate::{Bounds, FPair, Objective};

/// pyo3-exposed wrapper around `Bounds<f64>` constructed from numpy arrays.
#[pyclass(name = "Bounds")]
#[derive(Clone)]
pub struct PyBounds {
    pub(crate) inner: Bounds<f64>,
}

#[pymethods]
impl PyBounds {
    #[new]
    #[pyo3(signature = (low, high, slack = 1e-6))]
    fn new(
        low: PyReadonlyArray1<'_, f64>,
        high: PyReadonlyArray1<'_, f64>,
        slack: f64,
    ) -> PyResult<Self> {
        let low = Array1::from_vec(low.as_slice()?.to_vec());
        let high = Array1::from_vec(high.as_slice()?.to_vec());
        Ok(Self {
            inner: Bounds::new(low, high, slack),
        })
    }

    #[getter]
    fn dims(&self) -> usize {
        self.inner.dims
    }

    #[getter]
    fn slack(&self) -> f64 {
        self.inner.slack
    }

    fn contains<'py>(&self, x: PyReadonlyArray1<'py, f64>) -> PyResult<bool> {
        let slice = x.as_slice()?;
        Ok(self.inner.contains(ArrayView1::from(slice)))
    }

    fn clip<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let slice = x.as_slice()?;
        let clipped = self.inner.clip(ArrayView1::from(slice));
        Ok(PyArray1::from_slice(py, clipped.as_slice().expect("Array1 is contiguous")))
    }
}

/// pyo3-exposed wrapper around `FPair<f64>`.
#[pyclass(name = "FPair")]
#[derive(Clone)]
pub struct PyFPair {
    pub(crate) inner: FPair<f64>,
}

#[pymethods]
impl PyFPair {
    #[new]
    fn new(pos: PyReadonlyArray1<'_, f64>, val: f64) -> PyResult<Self> {
        let pos = Array1::from_vec(pos.as_slice()?.to_vec());
        Ok(Self {
            inner: FPair::new(pos, val),
        })
    }

    #[getter]
    fn pos<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(
            py,
            self.inner.pos.as_slice().expect("Array1 is contiguous"),
        )
    }

    #[getter]
    fn val(&self) -> f64 {
        self.inner.val
    }
}

/// Wraps a Python callable into the Rust `Objective<f64>` trait so the
/// SA driver loop can call user-defined Python objectives without leaving
/// Rust per evaluation.
#[pyclass(name = "PyObjective", unsendable)]
pub struct PyObjective {
    inner: Py<PyAny>,
    bounds: Bounds<f64>,
    dim: usize,
}

#[pymethods]
impl PyObjective {
    #[new]
    fn new(fn_: Py<PyAny>, bounds: PyBounds) -> Self {
        let dim = bounds.inner.dims;
        Self {
            inner: fn_,
            bounds: bounds.inner,
            dim,
        }
    }

    fn eval<'py>(&self, py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> PyResult<f64> {
        let slice = x.as_slice()?;
        let py_arr = PyArray1::from_slice(py, slice);
        let r = self.inner.call1(py, (py_arr,))?;
        r.extract::<f64>(py)
    }

    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }

    #[getter]
    fn bounds(&self) -> PyBounds {
        PyBounds {
            inner: self.bounds.clone(),
        }
    }
}

impl Objective<f64> for PyObjective {
    fn dim(&self) -> usize {
        self.dim
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        Python::attach(|py| {
            let owned: Vec<f64> = x.iter().copied().collect();
            let py_arr = PyArray1::from_vec(py, owned);
            let r = self
                .inner
                .call1(py, (py_arr,))
                .expect("PyObjective callable raised; surface this via PyErr instead in v0.4");
            r.extract::<f64>(py)
                .expect("PyObjective callable returned non-float; surface this in v0.4")
        })
    }
}
