//! The fallible path as a downstream crate sees it: an own error type, a
//! driver that returns it unchanged, and the blanket impls for infallible
//! types.

use eindir_core::{
    Bounds, DifferentiableObjective, Gradient, NanOnError, Objective, Rosenbrock,
    TryDifferentiableObjective, TryFiniteDiffGradient, TryGradient, TryObjective,
};
use ndarray::{Array1, ArrayView1, array};
use std::convert::Infallible;

const TOLERANCE: f64 = 1.0 / 1_000_000.0;

/// A caller's error, with data that a box or a string would lose.
#[derive(Debug, Clone, PartialEq)]
enum PesError {
    Refused { atom: usize, distance: f64 },
    Engine(i32),
}

struct Pes {
    bounds: Bounds<f64>,
}

impl Pes {
    fn new() -> Self {
        Self {
            bounds: Bounds::new(array![-5.0, -5.0], array![5.0, 5.0], 0.0),
        }
    }

    fn check(&self, x: ArrayView1<f64>) -> Result<(), PesError> {
        if x.iter().any(|v| !v.is_finite()) {
            return Err(PesError::Engine(-1));
        }
        if x[0] < -4.0 {
            return Err(PesError::Refused {
                atom: 0,
                distance: -x[0],
            });
        }
        Ok(())
    }
}

impl TryObjective<f64> for Pes {
    type Error = PesError;

    fn dims(&self) -> usize {
        2
    }

    fn domain(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn try_eval(&self, x: ArrayView1<f64>) -> Result<f64, PesError> {
        self.check(x)?;
        Ok(x.dot(&x))
    }
}

impl TryGradient<f64> for Pes {
    type GradError = PesError;

    fn try_grad(&self, x: ArrayView1<f64>) -> Result<Array1<f64>, PesError> {
        self.check(x)?;
        Ok(x.mapv(|v| 2.0 * v))
    }

    fn grad_dims(&self) -> usize {
        2
    }
}

impl TryDifferentiableObjective<f64> for Pes {}

/// The shape of a solver step: generic over the objective, returning the
/// objective's own error type.
fn descent_step<O>(obj: &O, x: &mut Array1<f64>, rate: f64) -> Result<f64, O::Error>
where
    O: TryDifferentiableObjective<f64> + ?Sized,
{
    let (value, grad) = obj.try_value_and_gradient(x.view())?;
    *x -= &(grad * rate);
    Ok(value)
}

#[test]
fn a_custom_error_reaches_the_caller_unchanged() {
    let pes = Pes::new();
    let mut x = array![1.0, 1.0];
    assert_eq!(descent_step(&pes, &mut x, 0.25), Ok(2.0));
    assert_eq!(x, array![0.5, 0.5]);

    let mut refused = array![-4.5, 0.0];
    assert_eq!(
        descent_step(&pes, &mut refused, 0.25),
        Err(PesError::Refused {
            atom: 0,
            distance: 4.5
        })
    );
    assert_eq!(refused, array![-4.5, 0.0], "a failed step leaves x alone");

    let mut broken = array![f64::NAN, 0.0];
    assert_eq!(
        descent_step(&pes, &mut broken, 0.25),
        Err(PesError::Engine(-1))
    );
}

#[test]
fn the_driver_works_through_a_trait_object() {
    let pes = Pes::new();
    let dynamic: &dyn TryDifferentiableObjective<f64, Error = PesError, GradError = PesError> =
        &pes;
    let mut x = array![2.0, 0.0];
    assert_eq!(descent_step(dynamic, &mut x, 0.5), Ok(4.0));
}

struct Bowl {
    bounds: Bounds<f64>,
}

impl Objective<f64> for Bowl {
    fn dim(&self) -> usize {
        2
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        x.dot(&x)
    }
}

impl Gradient<f64> for Bowl {
    fn dim(&self) -> usize {
        2
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        x.mapv(|v| 2.0 * v)
    }
}

impl DifferentiableObjective<f64> for Bowl {}

#[test]
fn an_infallible_type_is_accepted_by_the_same_driver() {
    let bowl = Bowl {
        bounds: Bounds::new(array![-5.0, -5.0], array![5.0, 5.0], 0.0),
    };
    let mut x = array![1.0, 1.0];
    let outcome: Result<f64, Infallible> = descent_step(&bowl, &mut x, 0.25);
    assert_eq!(outcome, Ok(2.0));
    assert_eq!(x, array![0.5, 0.5]);
}

#[test]
fn a_builtin_objective_is_a_try_objective() {
    let rosenbrock = Rosenbrock::<2>::new();
    let outcome: Result<f64, Infallible> = rosenbrock.try_eval(array![1.0, 1.0].view());
    assert_eq!(outcome, Ok(0.0));
}

#[test]
fn the_infallible_adapters_round_trip() {
    let adapted = NanOnError::new(Pes::new(), |_error: PesError| {});
    assert_eq!(adapted.eval(array![1.0, 2.0].view()), 5.0);
    assert!(adapted.eval(array![-4.5, 0.0].view()).is_nan());
    let back: Result<f64, Infallible> = adapted.try_eval(array![1.0, 2.0].view());
    assert_eq!(back, Ok(5.0));
}

#[test]
fn central_differences_over_a_fallible_objective_stop_at_the_error() {
    let fd = TryFiniteDiffGradient::new(Pes::new());
    let g = fd
        .try_grad(array![1.0, 2.0].view())
        .expect("interior point");
    assert!((g[0] - 2.0).abs() < TOLERANCE && (g[1] - 4.0).abs() < TOLERANCE);
    assert_eq!(
        fd.try_grad(array![-4.0, 0.0].view()),
        Err(PesError::Refused {
            atom: 0,
            distance: 4.0 + fd.step()
        })
    );
}

#[cfg(feature = "capi")]
mod capi {
    use super::*;
    use dlpk::sys::{
        DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLManagedTensorVersioned,
        DLPackVersion, DLTensor,
    };
    use eindir_core::ffi_try::{
        CallbackError, EINDIR_TRY_SUCCESS, EindirTryObjectiveWrapper, eindir_try_objective_free,
        eindir_try_objective_new,
    };
    use std::ffi::{c_char, c_void};

    const ENGINE_DOWN: i32 = 42;

    unsafe extern "C" fn flaky_eval(
        user_data: *mut c_void,
        x: *const DLManagedTensorVersioned,
        value_out: *mut f64,
        message_out: *mut c_char,
        message_cap: usize,
    ) -> i32 {
        let calls = unsafe { &mut *user_data.cast::<u32>() };
        *calls += 1;
        if *calls > 2 {
            let text = b"engine went away\0";
            assert!(message_cap >= text.len());
            unsafe { std::ptr::copy_nonoverlapping(text.as_ptr().cast(), message_out, text.len()) };
            return ENGINE_DOWN;
        }
        let t = unsafe { &(*x).dl_tensor };
        let n = unsafe { *t.shape } as usize;
        let xs = unsafe { std::slice::from_raw_parts(t.data as *const f64, n) };
        unsafe { *value_out = xs.iter().sum() };
        EINDIR_TRY_SUCCESS
    }

    #[test]
    fn a_callback_status_and_message_reach_a_generic_driver() {
        fn first_two_evals<O>(obj: &O) -> Result<Vec<f64>, O::Error>
        where
            O: TryObjective<f64> + ?Sized,
        {
            let a = obj.try_eval(array![1.0, 2.0].view())?;
            let b = obj.try_eval(array![3.0, 4.0].view())?;
            let c = obj.try_eval(array![5.0, 6.0].view())?;
            Ok(vec![a, b, c])
        }

        let mut calls = 0_u32;
        let mut low = [-9.0_f64; 2];
        let mut high = [9.0_f64; 2];
        let shape = [2_i64];
        let tensor = |data: &mut [f64; 2]| DLManagedTensorVersioned {
            version: DLPackVersion { major: 1, minor: 0 },
            manager_ctx: std::ptr::null_mut(),
            deleter: None,
            flags: 0,
            dl_tensor: DLTensor {
                data: data.as_mut_ptr().cast(),
                device: DLDevice {
                    device_type: DLDeviceType::kDLCPU,
                    device_id: 0,
                },
                ndim: 1,
                dtype: DLDataType {
                    code: DLDataTypeCode::kDLFloat,
                    bits: 64,
                    lanes: 1,
                },
                shape: shape.as_ptr().cast_mut(),
                strides: std::ptr::null_mut(),
                byte_offset: 0,
            },
        };
        let low_t = tensor(&mut low);
        let high_t = tensor(&mut high);
        let handle = unsafe {
            eindir_try_objective_new(
                2,
                &low_t,
                &high_t,
                flaky_eval,
                None,
                (&mut calls as *mut u32).cast(),
                None,
            )
        };
        assert!(!handle.is_null());
        let wrapper = unsafe { EindirTryObjectiveWrapper::new(&*handle) };
        let outcome = first_two_evals(&wrapper);
        assert_eq!(
            outcome,
            Err(CallbackError::new(ENGINE_DOWN, "engine went away"))
        );
        assert_eq!(calls, 3);
        unsafe { eindir_try_objective_free(handle) };
    }
}
