//! Adapter wrapping a Python callable into the Rust `Objective<f64>` trait.
//!
//! The numpy 0.26 crate is built against ndarray 0.16, while the rest of
//! eindir-core uses ndarray 0.17. To avoid an ABI mismatch between the two
//! ndarray copies, we cross the pyo3 boundary as `&[f64]` / `Vec<f64>` and
//! reconstruct ndarray 0.17 `Array1<f64>` / `ArrayView1<f64>` on the Rust
//! side via `Array1::from_vec` / `ArrayView1::from`.

use crate::{Bounds, DifferentiableObjective, FPair, Objective, gradient::Gradient};
use ndarray::{Array1, ArrayView1};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
        Ok(PyArray1::from_slice(
            py,
            clipped.as_slice().expect("Array1 is contiguous"),
        ))
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
        PyArray1::from_slice(py, self.inner.pos.as_slice().expect("Array1 is contiguous"))
    }

    #[getter]
    fn val(&self) -> f64 {
        self.inner.val
    }
}

/// Wraps a Python callable into the Rust `Objective<f64>` trait so the
/// SA driver loop can call user-defined Python objectives without leaving
/// Rust per evaluation.
///
/// When `grad_fn` is supplied at construction the object also implements
/// `Gradient<f64>` (native / analytic gradient, Ceres-style). Without a
/// supplied gradient, callers use `FiniteDiffGradient` explicitly.
#[pyclass(name = "PyObjective", unsendable)]
pub struct PyObjective {
    inner: Py<PyAny>,
    grad: Option<Py<PyAny>>,
    bounds: Bounds<f64>,
    dim: usize,
}

#[pymethods]
impl PyObjective {
    #[new]
    #[pyo3(signature = (fn_, bounds, grad_fn = None))]
    fn new(fn_: Py<PyAny>, bounds: PyBounds, grad_fn: Option<Py<PyAny>>) -> Self {
        let dim = bounds.inner.dims;
        Self {
            inner: fn_,
            grad: grad_fn,
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

    /// Native gradient (if one was supplied at construction).  Raises if
    /// this PyObjective has no grad_fn; callers should use
    /// FiniteDiffGradient in that case.
    fn grad<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let Some(g) = &self.grad else {
            return Err(PyValueError::new_err(
                "PyObjective was constructed without a grad_fn; supply grad_fn=... for native gradients (Ceres style) or wrap with FiniteDiffGradient",
            ));
        };
        let slice = x.as_slice()?;
        let py_arr = PyArray1::from_slice(py, slice);
        let r = g.call1(py, (py_arr,))?;
        r.extract::<Bound<'py, PyArray1<f64>>>(py)
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
                .expect("PyObjective callable raised");
            r.extract::<f64>(py)
                .expect("PyObjective callable returned non-float")
        })
    }
}

impl Gradient<f64> for PyObjective {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        if let Some(gfn) = &self.grad {
            Python::attach(|py| {
                let owned: Vec<f64> = x.iter().copied().collect();
                let py_arr = PyArray1::from_vec(py, owned);
                let r = gfn
                    .call1(py, (py_arr,))
                    .expect("PyObjective grad callable raised");
                let arr: Bound<PyArray1<f64>> = r
                    .extract(py)
                    .expect("grad callable must return array-like of f64");
                let ro = arr.readonly();
                Array1::from_vec(ro.as_slice().expect("contiguous").to_vec())
            })
        } else {
            panic!(
                "this PyObjective has no native gradient (constructed without grad_fn); \
                 use FiniteDiffGradient or supply one at construction (Ceres-style native grad)"
            );
        }
    }
    fn dim(&self) -> usize {
        self.dim
    }
}

impl DifferentiableObjective<f64> for PyObjective {}

/// Low-discrepancy points scaled to the supplied box bounds.
#[pyfunction]
#[pyo3(signature = (low, high, n, skip = 1))]
pub fn low_discrepancy_points(
    low: PyReadonlyArray1<'_, f64>,
    high: PyReadonlyArray1<'_, f64>,
    n: usize,
    skip: u64,
) -> PyResult<Vec<Vec<f64>>> {
    let low = low.as_slice()?.to_vec();
    let high = high.as_slice()?.to_vec();
    if low.len() != high.len() {
        return Err(PyValueError::new_err(
            "low and high must have the same length",
        ));
    }
    if low.is_empty() {
        return Err(PyValueError::new_err(
            "bounds must have at least one dimension",
        ));
    }
    if low.iter().zip(high.iter()).any(|(&lo, &hi)| hi < lo) {
        return Err(PyValueError::new_err(
            "each upper bound must be greater than or equal to the lower bound",
        ));
    }
    let bounds = Bounds::new(Array1::from_vec(low), Array1::from_vec(high), 0.0);
    let points = crate::pointset::low_discrepancy_points(&bounds, n, skip);
    Ok(points.outer_iter().map(|row| row.to_vec()).collect())
}

/// Deterministically shifted low-discrepancy points scaled to box bounds.
#[pyfunction]
#[pyo3(signature = (low, high, n, skip = 1, seed = 0))]
pub fn shifted_low_discrepancy_points(
    low: PyReadonlyArray1<'_, f64>,
    high: PyReadonlyArray1<'_, f64>,
    n: usize,
    skip: u64,
    seed: u64,
) -> PyResult<Vec<Vec<f64>>> {
    let low = low.as_slice()?.to_vec();
    let high = high.as_slice()?.to_vec();
    if low.len() != high.len() {
        return Err(PyValueError::new_err(
            "low and high must have the same length",
        ));
    }
    if low.is_empty() {
        return Err(PyValueError::new_err(
            "bounds must have at least one dimension",
        ));
    }
    if low.iter().zip(high.iter()).any(|(&lo, &hi)| hi < lo) {
        return Err(PyValueError::new_err(
            "each upper bound must be greater than or equal to the lower bound",
        ));
    }
    let bounds = Bounds::new(Array1::from_vec(low), Array1::from_vec(high), 0.0);
    let points = crate::pointset::shifted_low_discrepancy_points(&bounds, n, skip, seed);
    Ok(points.outer_iter().map(|row| row.to_vec()).collect())
}
