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

/// Box bounds ``[low, high]`` with a membership tolerance ``slack``.
///
/// ``low`` and ``high`` are contiguous 1-D float64 NumPy arrays of equal
/// length; other inputs raise ``TypeError``. ``slack`` (default ``1e-6``, in
/// the units of the coordinates) widens ``contains`` only.
///
/// Raises ``pyo3_runtime.PanicException``, which derives from
/// ``BaseException``, when the lengths differ, ``slack`` is negative or
/// ``low[i] > high[i]``.
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

    /// Number of dimensions.
    #[getter]
    fn dims(&self) -> usize {
        self.inner.dims
    }

    /// Membership tolerance used by ``contains``.
    #[getter]
    fn slack(&self) -> f64 {
        self.inner.slack
    }

    /// True when ``x`` has ``dims`` coordinates, each within ``slack`` of
    /// the box. A point of the wrong length gives ``False``.
    fn contains<'py>(&self, x: PyReadonlyArray1<'py, f64>) -> PyResult<bool> {
        let slice = x.as_slice()?;
        Ok(self.inner.contains(ArrayView1::from(slice)))
    }

    /// Returns a copy of ``x`` with each coordinate clipped to
    /// ``[low[i], high[i]]``, ignoring ``slack``.
    ///
    /// Raises ``PanicException`` when ``len(x) != dims``.
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

/// A position ``pos`` (1-D float64 array) paired with its value ``val``.
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

    /// The position, as a new NumPy array.
    #[getter]
    fn pos<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.pos.as_slice().expect("Array1 is contiguous"))
    }

    /// The objective value at ``pos``.
    #[getter]
    fn val(&self) -> f64 {
        self.inner.val
    }
}

/// A Python callable as an objective over ``bounds``.
///
/// ``fn_(x)`` takes a 1-D float64 NumPy array and returns a float.
/// ``grad_fn(x)``, when given, returns a 1-D float64 NumPy array of the same
/// length. Rust code sees the object as ``Objective<f64>`` and
/// ``Gradient<f64>``. Python has no finite-difference fallback: without
/// ``grad_fn``, ``grad`` raises ``ValueError``.
///
/// The object is bound to the thread that created it; using it from another
/// thread raises ``PanicException``.
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

    /// Calls ``fn_(x)`` and returns its result as a float.
    ///
    /// ``x`` must be a contiguous 1-D float64 array; its length is not
    /// checked against ``dim``. An exception raised by ``fn_`` propagates.
    fn eval<'py>(&self, py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> PyResult<f64> {
        let slice = x.as_slice()?;
        let py_arr = PyArray1::from_slice(py, slice);
        let r = self.inner.call1(py, (py_arr,))?;
        r.extract::<f64>(py)
    }

    /// Calls ``grad_fn(x)`` and returns its result.
    ///
    /// Raises ``ValueError`` when the object was built without ``grad_fn``.
    fn grad<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let Some(g) = &self.grad else {
            return Err(PyValueError::new_err(
                "PyObjective was constructed without a grad_fn; pass grad_fn=... to the constructor",
            ));
        };
        let slice = x.as_slice()?;
        let py_arr = PyArray1::from_slice(py, slice);
        let r = g.call1(py, (py_arr,))?;
        r.extract::<Bound<'py, PyArray1<f64>>>(py)
    }

    /// Number of input dimensions, equal to ``bounds.dims``.
    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }

    /// A copy of the bounds.
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

/// Halton points scaled to the box ``[low, high]``, as a list of rows.
///
/// ``low`` and ``high`` are contiguous 1-D float64 arrays. Row ``r`` is
/// Halton point ``skip + r``; ``skip=1`` (the default) avoids the corner
/// ``low``. ``eindir.low_discrepancy_points`` wraps this and returns an
/// ``(n, dims)`` array.
///
/// # Errors
///
/// Raises ``ValueError`` when the lengths differ, the arrays are empty or
/// ``high[i] < low[i]``, and ``TypeError`` for a non-contiguous or non-float64
/// array.
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

/// Halton points with a seeded Cranley-Patterson shift, scaled to the box.
///
/// The same ``seed`` gives the same points for a given build. Arguments as
/// in ``low_discrepancy_points``; ``seed`` defaults to 0.
///
/// # Errors
///
/// As ``low_discrepancy_points``.
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
