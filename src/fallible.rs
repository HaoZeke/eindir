//! Fallible objectives and gradients.
//!
//! [`Objective::eval`] and [`Gradient::grad`] cannot fail, so a surface that
//! can (an engine error, a non-finite input, a refused geometry) has to panic
//! or return a sentinel. The traits here add a second, fallible path next to
//! the first without changing it:
//!
//! - [`TryObjective`], [`TryGradient`] and [`TryDifferentiableObjective`]
//!   carry an associated `Error` type and `try_*` methods.
//! - Every [`Objective`], [`Gradient`] and
//!   [`DifferentiableObjective`] is also the matching `Try*` trait through a
//!   blanket impl, with `Error = core::convert::Infallible`.
//! - [`NanOnError`] and [`PanicOnError`] go the other way, from a `Try*` type
//!   to the infallible traits.
//! - [`TryOracle`] and [`TryFiniteDiffGradient`] build fallible objectives
//!   from a closure and from a value-only objective.
//!
//! The error type is the implementor's own. Nothing here boxes it, converts
//! it or wraps it, so a driver that is generic over `O: TryObjective<f64>`
//! returns `O::Error` exactly as the objective produced it.
//!
//! The `Try*` traits name their domain methods [`TryObjective::dims`],
//! [`TryObjective::domain`] and [`TryGradient::grad_dims`], not `dim` and
//! `bounds`. Because of the blanket impls, an infallible type is also a
//! `Try*` type, and equal names would make `x.dim()` ambiguous (E0034) as
//! soon as both traits are in scope.

use crate::{Bounds, DifferentiableObjective, FPair, Gradient, Objective};
use core::convert::Infallible;
use ndarray::{Array1, ArrayView1, ArrayView2};
use num_traits::Float;
use rayon::prelude::*;
use std::fmt::Debug;
use std::marker::PhantomData;

/// A real-valued function on `R^dims` whose evaluation can fail.
///
/// The fallible counterpart of [`Objective`]. Implement it directly for a
/// surface that can fail. Every [`Objective`] already implements it with
/// `Error = Infallible`, so a driver written against `TryObjective` accepts
/// both. A type that implements `Objective` must not implement this trait
/// by hand: the blanket impl already covers it.
pub trait TryObjective<T: Float>: Send + Sync {
    /// The failure of one evaluation, passed through unchanged.
    type Error;

    /// Number of input dimensions; matches `domain().dims`.
    fn dims(&self) -> usize;

    /// The feasible domain (typically a box).
    fn domain(&self) -> &Bounds<T>;

    /// Evaluates the objective at a single point.
    ///
    /// # Errors
    ///
    /// Returns the implementor's error when the evaluation fails.
    fn try_eval(&self, x: ArrayView1<T>) -> Result<T, Self::Error>;

    /// Evaluates a batch of points (rows of `x`).
    ///
    /// Default: serial loop over [`TryObjective::try_eval`] that stops at the
    /// first error.
    ///
    /// # Errors
    ///
    /// Returns the error of the first failing row.
    fn try_eval_batch(&self, x: ArrayView2<T>) -> Result<Array1<T>, Self::Error> {
        x.outer_iter().map(|row| self.try_eval(row)).collect()
    }

    /// Optional known global minimum. Default returns `None`.
    fn known_min(&self) -> Option<&FPair<T>> {
        None
    }
}

/// First-derivative interface whose evaluation can fail.
///
/// The fallible counterpart of [`Gradient`], with the same blanket impl
/// and `Error = Infallible` for every [`Gradient`].
pub trait TryGradient<T: Float>: Send + Sync {
    /// The failure of one gradient evaluation, passed through unchanged.
    ///
    /// Named `GradError`, not `Error`, so that `O::Error` stays unambiguous
    /// for a driver bounded by [`TryDifferentiableObjective`], which has both
    /// traits as supertraits and equates the two.
    type GradError;

    /// Gradient of the underlying scalar field at `x`.
    ///
    /// # Errors
    ///
    /// Returns the implementor's error when the evaluation fails.
    fn try_grad(&self, x: ArrayView1<T>) -> Result<Array1<T>, Self::GradError>;

    /// Number of input dimensions (must match the paired
    /// [`TryObjective::dims`]).
    fn grad_dims(&self) -> usize;
}

/// A [`TryObjective`] that also supplies its gradient, with one error type.
///
/// The two `Error` types are the same, so `O::Error` in a driver bounded by
/// `O: TryDifferentiableObjective<f64>` is unambiguous. Every
/// [`DifferentiableObjective`] implements it through a blanket impl, and
/// keeps its fused [`DifferentiableObjective::value_and_gradient`]. For a
/// fallible type, write an empty
/// `impl TryDifferentiableObjective<f64> for T {}`, or override
/// [`TryDifferentiableObjective::try_value_and_gradient`] with a fused
/// evaluation.
pub trait TryDifferentiableObjective<T: Float>:
    TryObjective<T> + TryGradient<T, GradError = <Self as TryObjective<T>>::Error>
{
    /// Returns `(f(x), ∇f(x))`. Default calls the two methods separately and
    /// stops at the first error.
    ///
    /// # Errors
    ///
    /// Returns the error of whichever evaluation fails first.
    fn try_value_and_gradient(
        &self,
        x: ArrayView1<T>,
    ) -> Result<(T, Array1<T>), <Self as TryObjective<T>>::Error> {
        let value = self.try_eval(x)?;
        let grad = self.try_grad(x)?;
        Ok((value, grad))
    }
}

impl<T, O> TryObjective<T> for O
where
    T: Float,
    O: Objective<T> + ?Sized,
{
    type Error = Infallible;

    fn dims(&self) -> usize {
        Objective::dim(self)
    }

    fn domain(&self) -> &Bounds<T> {
        self.bounds()
    }

    fn try_eval(&self, x: ArrayView1<T>) -> Result<T, Infallible> {
        Ok(self.eval(x))
    }

    fn try_eval_batch(&self, x: ArrayView2<T>) -> Result<Array1<T>, Infallible> {
        Ok(self.eval_batch(x))
    }

    fn known_min(&self) -> Option<&FPair<T>> {
        self.global_min()
    }
}

impl<T, G> TryGradient<T> for G
where
    T: Float,
    G: Gradient<T> + ?Sized,
{
    type GradError = Infallible;

    fn try_grad(&self, x: ArrayView1<T>) -> Result<Array1<T>, Infallible> {
        Ok(self.grad(x))
    }

    fn grad_dims(&self) -> usize {
        Gradient::dim(self)
    }
}

impl<T, O> TryDifferentiableObjective<T> for O
where
    T: Float,
    O: DifferentiableObjective<T> + ?Sized,
{
    fn try_value_and_gradient(&self, x: ArrayView1<T>) -> Result<(T, Array1<T>), Infallible> {
        Ok(self.value_and_gradient(x))
    }
}

/// Parallel batch evaluation for a [`TryObjective<f64>`].
///
/// Uses Rayon when `x.nrows() >= EVAL_BATCH_PARALLEL_MIN`, and
/// [`TryObjective::try_eval_batch`] below that. When several rows fail it
/// returns the error of the lowest-numbered one, so the result does not
/// depend on thread scheduling.
///
/// # Errors
///
/// Returns the error of the first failing row.
pub fn try_eval_batch_parallel<O>(
    obj: &O,
    x: ArrayView2<f64>,
) -> Result<Array1<f64>, <O as TryObjective<f64>>::Error>
where
    O: TryObjective<f64> + ?Sized,
    <O as TryObjective<f64>>::Error: Send,
{
    let n = x.nrows();
    if n < crate::EVAL_BATCH_PARALLEL_MIN {
        return obj.try_eval_batch(x);
    }
    let rows: Vec<Result<f64, <O as TryObjective<f64>>::Error>> = (0..n)
        .into_par_iter()
        .map(|i| obj.try_eval(x.row(i)))
        .collect();
    rows.into_iter().collect()
}

/// A [`TryObjective`] and [`TryGradient`] back to [`Objective`] and
/// [`Gradient`], reporting a failure as NaN.
///
/// A failed value is `T::nan()` and a failed gradient is a NaN vector of
/// length [`TryGradient::grad_dims`]. Each error goes to the `sink` first,
/// so it is not lost: log it, count it, or keep the first one.
/// This is the sentinel convention of the C wrapper, with the error kept.
pub struct NanOnError<O, F> {
    inner: O,
    sink: F,
}

impl<O, F> NanOnError<O, F> {
    /// Wraps `inner`, handing every error to `sink` before returning NaN.
    pub fn new(inner: O, sink: F) -> Self {
        Self { inner, sink }
    }

    /// The wrapped objective.
    pub fn inner(&self) -> &O {
        &self.inner
    }

    /// Returns the wrapped objective.
    pub fn into_inner(self) -> O {
        self.inner
    }
}

impl<T, O, F> Objective<T> for NanOnError<O, F>
where
    T: Float,
    O: TryObjective<T>,
    F: Fn(<O as TryObjective<T>>::Error) + Send + Sync,
{
    fn dim(&self) -> usize {
        self.inner.dims()
    }

    fn bounds(&self) -> &Bounds<T> {
        self.inner.domain()
    }

    fn eval(&self, x: ArrayView1<T>) -> T {
        match self.inner.try_eval(x) {
            Ok(value) => value,
            Err(error) => {
                (self.sink)(error);
                T::nan()
            }
        }
    }

    fn global_min(&self) -> Option<&FPair<T>> {
        self.inner.known_min()
    }
}

impl<T, O, F> Gradient<T> for NanOnError<O, F>
where
    T: Float,
    O: TryGradient<T>,
    F: Fn(<O as TryGradient<T>>::GradError) + Send + Sync,
{
    fn grad(&self, x: ArrayView1<T>) -> Array1<T> {
        match self.inner.try_grad(x) {
            Ok(grad) => grad,
            Err(error) => {
                (self.sink)(error);
                Array1::from_elem(self.inner.grad_dims(), T::nan())
            }
        }
    }

    fn dim(&self) -> usize {
        self.inner.grad_dims()
    }
}

impl<T, O, F> DifferentiableObjective<T> for NanOnError<O, F>
where
    T: Float,
    O: TryDifferentiableObjective<T>,
    F: Fn(<O as TryObjective<T>>::Error) + Send + Sync,
{
    fn value_and_gradient(&self, x: ArrayView1<T>) -> (T, Array1<T>) {
        match self.inner.try_value_and_gradient(x) {
            Ok(pair) => pair,
            Err(error) => {
                (self.sink)(error);
                (
                    T::nan(),
                    Array1::from_elem(self.inner.grad_dims(), T::nan()),
                )
            }
        }
    }
}

/// A [`TryObjective`] and [`TryGradient`] back to [`Objective`] and
/// [`Gradient`], panicking on a failure.
///
/// For tests and scripts, where a failure is a bug. The panic message is the
/// `Debug` form of the error.
pub struct PanicOnError<O> {
    inner: O,
}

impl<O> PanicOnError<O> {
    /// Wraps `inner`.
    pub fn new(inner: O) -> Self {
        Self { inner }
    }

    /// The wrapped objective.
    pub fn inner(&self) -> &O {
        &self.inner
    }

    /// Returns the wrapped objective.
    pub fn into_inner(self) -> O {
        self.inner
    }
}

impl<T, O> Objective<T> for PanicOnError<O>
where
    T: Float,
    O: TryObjective<T>,
    <O as TryObjective<T>>::Error: Debug,
{
    fn dim(&self) -> usize {
        self.inner.dims()
    }

    fn bounds(&self) -> &Bounds<T> {
        self.inner.domain()
    }

    fn eval(&self, x: ArrayView1<T>) -> T {
        match self.inner.try_eval(x) {
            Ok(value) => value,
            Err(error) => panic!("objective evaluation failed: {error:?}"),
        }
    }

    fn global_min(&self) -> Option<&FPair<T>> {
        self.inner.known_min()
    }
}

impl<T, O> Gradient<T> for PanicOnError<O>
where
    T: Float,
    O: TryGradient<T>,
    <O as TryGradient<T>>::GradError: Debug,
{
    fn grad(&self, x: ArrayView1<T>) -> Array1<T> {
        match self.inner.try_grad(x) {
            Ok(grad) => grad,
            Err(error) => panic!("gradient evaluation failed: {error:?}"),
        }
    }

    fn dim(&self) -> usize {
        self.inner.grad_dims()
    }
}

impl<T, O> DifferentiableObjective<T> for PanicOnError<O>
where
    T: Float,
    O: TryDifferentiableObjective<T>,
    <O as TryObjective<T>>::Error: Debug,
{
    fn value_and_gradient(&self, x: ArrayView1<T>) -> (T, Array1<T>) {
        match self.inner.try_value_and_gradient(x) {
            Ok(pair) => pair,
            Err(error) => panic!("value and gradient evaluation failed: {error:?}"),
        }
    }
}

/// A closure `x -> Result<(f, ∇f), E>` as a [`TryDifferentiableObjective`].
///
/// The fallible counterpart of a fused value-and-gradient oracle: one call
/// produces both, so [`TryDifferentiableObjective::try_value_and_gradient`]
/// costs one engine call, and `try_eval` and `try_grad` each cost one and
/// drop the half they do not need.
pub struct TryOracle<F, E> {
    f: F,
    bounds: Bounds<f64>,
    error: PhantomData<fn() -> E>,
}

impl<F, E> TryOracle<F, E>
where
    F: Fn(ArrayView1<f64>) -> Result<(f64, Array1<f64>), E> + Send + Sync,
{
    /// Builds an oracle over `bounds` from `f`.
    pub fn new(bounds: Bounds<f64>, f: F) -> Self {
        Self {
            f,
            bounds,
            error: PhantomData,
        }
    }
}

impl<F, E> TryObjective<f64> for TryOracle<F, E>
where
    F: Fn(ArrayView1<f64>) -> Result<(f64, Array1<f64>), E> + Send + Sync,
{
    type Error = E;

    fn dims(&self) -> usize {
        self.bounds.dims
    }

    fn domain(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn try_eval(&self, x: ArrayView1<f64>) -> Result<f64, E> {
        (self.f)(x).map(|(value, _)| value)
    }
}

impl<F, E> TryGradient<f64> for TryOracle<F, E>
where
    F: Fn(ArrayView1<f64>) -> Result<(f64, Array1<f64>), E> + Send + Sync,
{
    type GradError = E;

    fn try_grad(&self, x: ArrayView1<f64>) -> Result<Array1<f64>, E> {
        (self.f)(x).map(|(_, grad)| grad)
    }

    fn grad_dims(&self) -> usize {
        self.bounds.dims
    }
}

impl<F, E> TryDifferentiableObjective<f64> for TryOracle<F, E>
where
    F: Fn(ArrayView1<f64>) -> Result<(f64, Array1<f64>), E> + Send + Sync,
{
    fn try_value_and_gradient(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), E> {
        (self.f)(x)
    }
}

/// Central-difference gradient of any [`TryObjective<f64>`], propagating the
/// first evaluation error.
///
/// The stencil is that of [`crate::FiniteDiffGradient`]: `2 * dims`
/// evaluations per gradient. The wrapped objective keeps its own error type.
pub struct TryFiniteDiffGradient<O> {
    obj: O,
    h: f64,
}

impl<O: TryObjective<f64>> TryFiniteDiffGradient<O> {
    /// Step used by [`TryFiniteDiffGradient::new`], the same as
    /// [`crate::FiniteDiffGradient::new`].
    pub const DEFAULT_STEP: f64 = 1.0 / 100_000.0;

    /// Wraps `obj` with the default step.
    pub fn new(obj: O) -> Self {
        Self {
            obj,
            h: Self::DEFAULT_STEP,
        }
    }

    /// Sets the symmetric perturbation step.
    ///
    /// # Panics
    ///
    /// Panics when `h` is not positive and finite.
    #[must_use]
    pub fn with_step(mut self, h: f64) -> Self {
        assert!(h > 0.0 && h.is_finite(), "h must be positive and finite");
        self.h = h;
        self
    }

    /// The symmetric perturbation step.
    pub fn step(&self) -> f64 {
        self.h
    }

    /// The wrapped objective.
    pub fn inner(&self) -> &O {
        &self.obj
    }

    /// Returns the wrapped objective.
    pub fn into_inner(self) -> O {
        self.obj
    }
}

impl<O: TryObjective<f64>> TryGradient<f64> for TryFiniteDiffGradient<O> {
    type GradError = <O as TryObjective<f64>>::Error;

    fn try_grad(&self, x: ArrayView1<f64>) -> Result<Array1<f64>, Self::GradError> {
        let n = x.len();
        let mut g = Array1::zeros(n);
        let mut xp = x.to_owned();
        let mut xm = x.to_owned();
        for i in 0..n {
            xp[i] = x[i] + self.h;
            xm[i] = x[i] - self.h;
            let hi = self.obj.try_eval(xp.view())?;
            let lo = self.obj.try_eval(xm.view())?;
            g[i] = (hi - lo) / (2.0 * self.h);
            xp[i] = x[i];
            xm[i] = x[i];
        }
        Ok(g)
    }

    fn grad_dims(&self) -> usize {
        self.obj.dims()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::sync::Mutex;

    const TOLERANCE: f64 = 1.0 / 1_000_000.0;

    #[derive(Debug, Clone, PartialEq)]
    enum EngineError {
        Refused { atom: usize },
        NonFinite,
    }

    /// `f(x) = sum x_i^2`, refusing any point with a coordinate above 10 and
    /// any non-finite coordinate.
    struct Engine {
        bounds: Bounds<f64>,
    }

    impl Engine {
        fn new(dims: usize) -> Self {
            Self {
                bounds: Bounds::new(
                    Array1::from_elem(dims, -20.0),
                    Array1::from_elem(dims, 20.0),
                    0.0,
                ),
            }
        }

        fn check(x: ArrayView1<f64>) -> Result<(), EngineError> {
            if x.iter().any(|v| !v.is_finite()) {
                return Err(EngineError::NonFinite);
            }
            match x.iter().position(|&v| v > 10.0) {
                Some(atom) => Err(EngineError::Refused { atom }),
                None => Ok(()),
            }
        }
    }

    impl TryObjective<f64> for Engine {
        type Error = EngineError;

        fn dims(&self) -> usize {
            self.bounds.dims
        }

        fn domain(&self) -> &Bounds<f64> {
            &self.bounds
        }

        fn try_eval(&self, x: ArrayView1<f64>) -> Result<f64, EngineError> {
            Self::check(x)?;
            Ok(x.dot(&x))
        }
    }

    impl TryGradient<f64> for Engine {
        type GradError = EngineError;

        fn try_grad(&self, x: ArrayView1<f64>) -> Result<Array1<f64>, EngineError> {
            Self::check(x)?;
            Ok(x.mapv(|v| 2.0 * v))
        }

        fn grad_dims(&self) -> usize {
            self.bounds.dims
        }
    }

    impl TryDifferentiableObjective<f64> for Engine {}

    /// Generic over the error, the way a driver is.
    fn run<O>(obj: &O, x: &Array1<f64>) -> Result<f64, O::Error>
    where
        O: TryDifferentiableObjective<f64> + ?Sized,
    {
        let (value, grad) = obj.try_value_and_gradient(x.view())?;
        Ok(value + grad.sum())
    }

    #[test]
    fn custom_error_passes_through_unchanged() {
        let engine = Engine::new(3);
        assert_eq!(engine.try_eval(array![1.0, 2.0, 3.0].view()), Ok(14.0));
        assert_eq!(
            engine.try_eval(array![1.0, 11.0, 3.0].view()),
            Err(EngineError::Refused { atom: 1 })
        );
        assert_eq!(
            engine.try_grad(array![f64::NAN, 0.0, 0.0].view()),
            Err(EngineError::NonFinite)
        );
        assert_eq!(run(&engine, &array![1.0, 2.0, 3.0]), Ok(14.0 + 12.0));
        assert_eq!(
            run(&engine, &array![0.0, 0.0, 12.0]),
            Err(EngineError::Refused { atom: 2 })
        );
    }

    #[test]
    fn batch_stops_at_the_first_failing_row() {
        let engine = Engine::new(2);
        let ok = array![[1.0, 1.0], [2.0, 2.0]];
        assert_eq!(engine.try_eval_batch(ok.view()), Ok(array![2.0, 8.0]));
        let bad = array![[1.0, 1.0], [11.0, 0.0], [f64::NAN, 0.0]];
        assert_eq!(
            engine.try_eval_batch(bad.view()),
            Err(EngineError::Refused { atom: 0 })
        );
    }

    #[test]
    fn parallel_batch_reports_the_lowest_failing_row() {
        let engine = Engine::new(1);
        let n = 4 * crate::EVAL_BATCH_PARALLEL_MIN;
        let mut x = Array1::from_iter((0..n).map(|i| i as f64 / n as f64))
            .into_shape_with_order((n, 1))
            .expect("column of n rows");
        assert_eq!(
            try_eval_batch_parallel(&engine, x.view()).map(|v| v.len()),
            Ok(n)
        );
        x[[7, 0]] = f64::NAN;
        x[[n - 1, 0]] = 11.0;
        assert_eq!(
            try_eval_batch_parallel(&engine, x.view()),
            Err(EngineError::NonFinite)
        );
    }

    /// An ordinary, infallible type: `Objective + Gradient`.
    struct Quadratic {
        bounds: Bounds<f64>,
    }

    impl Objective<f64> for Quadratic {
        fn dim(&self) -> usize {
            self.bounds.dims
        }
        fn bounds(&self) -> &Bounds<f64> {
            &self.bounds
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            x.dot(&x)
        }
    }

    impl Gradient<f64> for Quadratic {
        fn dim(&self) -> usize {
            self.bounds.dims
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            x.mapv(|v| 2.0 * v)
        }
    }

    impl DifferentiableObjective<f64> for Quadratic {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (x.dot(&x), x.mapv(|v| 2.0 * v))
        }
    }

    fn quadratic() -> Quadratic {
        Quadratic {
            bounds: Bounds::new(Array1::from_elem(2, -5.0), Array1::from_elem(2, 5.0), 0.0),
        }
    }

    #[test]
    fn infallible_types_are_try_types_with_infallible_error() {
        let q = quadratic();
        let x = array![1.0, 2.0];
        let value: Result<f64, Infallible> = q.try_eval(x.view());
        assert_eq!(value, Ok(5.0));
        assert_eq!(q.try_grad(x.view()), Ok(array![2.0, 4.0]));
        assert_eq!(q.dims(), 2);
        assert_eq!(q.grad_dims(), 2);
        assert_eq!(q.domain().dims, 2);
        assert_eq!(run(&q, &x), Ok(5.0 + 6.0));
        let batch = array![[1.0, 0.0], [0.0, 3.0]];
        assert_eq!(q.try_eval_batch(batch.view()), Ok(array![1.0, 9.0]));
        assert_eq!(
            try_eval_batch_parallel(&q, batch.view()),
            Ok(array![1.0, 9.0])
        );
    }

    #[test]
    fn builtin_objectives_work_through_the_blanket_impl() {
        let sphere = crate::Rastrigin::<2>::new();
        let x = array![0.0, 0.0];
        assert_eq!(sphere.try_eval(x.view()), Ok(0.0));
        assert!(sphere.known_min().is_some());
        let dyn_obj: &dyn Objective<f64> = &sphere;
        assert_eq!(dyn_obj.try_eval(x.view()), Ok(0.0));
    }

    #[test]
    fn infallible_try_value_and_gradient_keeps_the_fused_override() {
        struct Fused(Bounds<f64>);
        impl Objective<f64> for Fused {
            fn dim(&self) -> usize {
                1
            }
            fn bounds(&self) -> &Bounds<f64> {
                &self.0
            }
            fn eval(&self, _x: ArrayView1<f64>) -> f64 {
                panic!("split eval must not run")
            }
        }
        impl Gradient<f64> for Fused {
            fn dim(&self) -> usize {
                1
            }
            fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> {
                panic!("split grad must not run")
            }
        }
        impl DifferentiableObjective<f64> for Fused {
            fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
                (x[0], array![1.0])
            }
        }
        let fused = Fused(Bounds::new(array![-1.0], array![1.0], 0.0));
        assert_eq!(
            fused.try_value_and_gradient(array![0.5].view()),
            Ok((0.5, array![1.0]))
        );
    }

    #[test]
    fn nan_on_error_returns_nan_and_reports_the_error() {
        let seen = Mutex::new(Vec::new());
        let wrapped = NanOnError::new(Engine::new(2), |e: EngineError| {
            seen.lock().expect("sink lock").push(e);
        });
        assert_eq!(wrapped.eval(array![1.0, 2.0].view()), 5.0);
        assert!(wrapped.eval(array![11.0, 0.0].view()).is_nan());
        let g = wrapped.grad(array![f64::NAN, 0.0].view());
        assert_eq!(g.len(), 2);
        assert!(g.iter().all(|v| v.is_nan()));
        let (v, g) = wrapped.value_and_gradient(array![0.0, 12.0].view());
        assert!(v.is_nan() && g.iter().all(|v| v.is_nan()));
        assert_eq!(
            *seen.lock().expect("sink lock"),
            vec![
                EngineError::Refused { atom: 0 },
                EngineError::NonFinite,
                EngineError::Refused { atom: 1 },
            ]
        );
        assert_eq!(Objective::dim(&wrapped), 2);
        assert_eq!(Gradient::dim(&wrapped), 2);
    }

    #[test]
    fn panic_on_error_passes_values_and_panics_on_failure() {
        let wrapped = PanicOnError::new(Engine::new(2));
        assert_eq!(wrapped.eval(array![1.0, 2.0].view()), 5.0);
        assert_eq!(wrapped.grad(array![1.0, 2.0].view()), array![2.0, 4.0]);
        let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            wrapped.eval(array![11.0, 0.0].view())
        }))
        .expect_err("a refused point must panic");
        let message = failure
            .downcast_ref::<String>()
            .expect("a formatted panic message");
        assert!(message.contains("Refused"), "{message}");
    }

    struct GradFailure {
        bounds: Bounds<f64>,
    }

    impl TryObjective<f64> for GradFailure {
        type Error = EngineError;

        fn dims(&self) -> usize {
            self.bounds.dims
        }

        fn domain(&self) -> &Bounds<f64> {
            &self.bounds
        }

        fn try_eval(&self, x: ArrayView1<f64>) -> Result<f64, EngineError> {
            Ok(x.dot(&x))
        }
    }

    impl TryGradient<f64> for GradFailure {
        type GradError = EngineError;

        fn try_grad(&self, _x: ArrayView1<f64>) -> Result<Array1<f64>, EngineError> {
            Err(EngineError::NonFinite)
        }

        fn grad_dims(&self) -> usize {
            self.bounds.dims
        }
    }

    impl TryDifferentiableObjective<f64> for GradFailure {}

    #[test]
    fn panic_on_error_distinguishes_value_and_gradient_failures() {
        let engine = PanicOnError::new(Engine::new(2));
        let eval_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine.eval(array![11.0, 0.0].view())
        }))
        .expect_err("eval failure must panic");
        let eval_msg = eval_panic
            .downcast_ref::<String>()
            .expect("formatted panic message");
        assert!(
            eval_msg.starts_with("objective evaluation failed: ") && eval_msg.contains("Refused"),
            "{eval_msg}"
        );

        let grad_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine.grad(array![11.0, 0.0].view())
        }))
        .expect_err("grad failure must panic");
        let grad_msg = grad_panic
            .downcast_ref::<String>()
            .expect("formatted panic message");
        assert!(
            grad_msg.starts_with("gradient evaluation failed: ") && grad_msg.contains("Refused"),
            "{grad_msg}"
        );

        let grad_only = PanicOnError::new(GradFailure {
            bounds: Bounds::new(array![-5.0, -5.0], array![5.0, 5.0], 0.0),
        });
        let vg_grad_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            grad_only.value_and_gradient(array![1.0, 2.0].view())
        }))
        .expect_err("gradient failure inside value_and_gradient must panic");
        let vg_grad_msg = vg_grad_panic
            .downcast_ref::<String>()
            .expect("formatted panic message");
        assert!(
            vg_grad_msg.starts_with("value and gradient evaluation failed: ")
                && vg_grad_msg.contains("NonFinite"),
            "{vg_grad_msg}"
        );
        assert!(
            !vg_grad_msg.contains("objective evaluation failed"),
            "{vg_grad_msg}"
        );

        let vg_eval_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine.value_and_gradient(array![11.0, 0.0].view())
        }))
        .expect_err("value failure inside value_and_gradient must panic");
        let vg_eval_msg = vg_eval_panic
            .downcast_ref::<String>()
            .expect("formatted panic message");
        assert!(
            vg_eval_msg.starts_with("value and gradient evaluation failed: ")
                && vg_eval_msg.contains("Refused"),
            "{vg_eval_msg}"
        );
    }

    #[test]
    fn try_oracle_is_fused_and_propagates() {
        let oracle = TryOracle::new(
            Bounds::new(Array1::from_elem(2, -5.0), Array1::from_elem(2, 5.0), 0.0),
            |x: ArrayView1<f64>| {
                if x[0] > 1.0 {
                    Err(EngineError::Refused { atom: 0 })
                } else {
                    Ok((x.dot(&x), x.mapv(|v| 2.0 * v)))
                }
            },
        );
        assert_eq!(
            oracle.try_value_and_gradient(array![1.0, 1.0].view()),
            Ok((2.0, array![2.0, 2.0]))
        );
        assert_eq!(oracle.try_eval(array![0.5, 0.0].view()), Ok(0.25));
        assert_eq!(
            oracle.try_grad(array![2.0, 0.0].view()),
            Err(EngineError::Refused { atom: 0 })
        );
        assert_eq!(run(&oracle, &array![0.5, 0.5]), Ok(0.5 + 2.0));
    }

    #[test]
    fn try_finite_diff_matches_the_gradient_and_propagates_errors() {
        let fd = TryFiniteDiffGradient::new(Engine::new(2));
        let g = fd.try_grad(array![1.0, -2.0].view()).expect("finite point");
        assert!((g[0] - 2.0).abs() < TOLERANCE && (g[1] + 4.0).abs() < TOLERANCE);
        assert_eq!(
            fd.try_grad(array![10.0, 0.0].view()),
            Err(EngineError::Refused { atom: 0 })
        );
        assert_eq!(fd.grad_dims(), 2);
        let coarse = TryFiniteDiffGradient::new(Engine::new(2)).with_step(0.5);
        assert_eq!(coarse.step(), 0.5);
        let plain = crate::FiniteDiffGradient::new(quadratic());
        let lifted = TryFiniteDiffGradient::new(quadratic());
        let x = array![0.3, -0.7];
        assert_eq!(
            lifted.try_grad(x.view()).expect("infallible"),
            plain.grad(x.view())
        );
    }

    #[test]
    #[should_panic(expected = "h must be positive and finite")]
    fn try_finite_diff_rejects_a_bad_step() {
        let _ = TryFiniteDiffGradient::new(Engine::new(1)).with_step(0.0);
    }
}
