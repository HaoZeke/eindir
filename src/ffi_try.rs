//! Fallible C callbacks: a status code plus a message.
//!
//! The callbacks of `eindir_objective_t` return one of three
//! `eindir_status_t` values and cannot say why, so a Rust driver sees a
//! failure only as a NaN.
//! The handle here, [`eindir_try_objective_t`], takes callbacks that return
//! an `int32_t` status and write a NUL-terminated message into a buffer the
//! library lends them. The old handle, its callbacks and its layout are
//! unchanged; this one lives beside it.
//!
//! Status `0` is success and any other value is a failure. `1` and `2` mean
//! the same as `EINDIR_INVALID_PARAMETER` and `EINDIR_INTERNAL_ERROR`, and a
//! callback may use any other value for its own codes. The status and the
//! message reach Rust as a [`CallbackError`] through the
//! [`TryObjective`] impl of
//! [`EindirTryObjectiveWrapper`], and reach C as the return value and the
//! buffer of `eindir_try_objective_eval` and `eindir_try_objective_grad`.

use std::ffi::{c_char, c_void};

use dlpk::sys::DLManagedTensorVersioned;
use ndarray::{Array1, ArrayView1};
use thiserror::Error;

use crate::ffi::{copy_bounds, create_borrowed_f64_1d, set_last_error, tensor_free};
use crate::{Bounds, TryDifferentiableObjective, TryGradient, TryObjective};

/// Status of a fallible callback or entry point that succeeded.
pub const EINDIR_TRY_SUCCESS: i32 = 0;
/// Status for a NULL or otherwise unusable argument; equals
/// `EINDIR_INVALID_PARAMETER`.
pub const EINDIR_TRY_INVALID_PARAMETER: i32 = 1;
/// Status for an internal failure or a panic; equals
/// `EINDIR_INTERNAL_ERROR`.
pub const EINDIR_TRY_INTERNAL_ERROR: i32 = 2;
/// Size in bytes of the message buffer lent to every fallible callback,
/// including the terminating NUL.
pub const EINDIR_TRY_MESSAGE_CAP: usize = 256;

/// Fallible callback for evaluating the objective value.
///
/// Returns `0` on success. On failure it returns a nonzero status and should
/// write a NUL-terminated message of at most `message_cap - 1` bytes to
/// `message_out`. `message_out` is never NULL and `message_cap` is
/// [`EINDIR_TRY_MESSAGE_CAP`] when eindir calls it.
pub type EindirTryEvalFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
    message_out: *mut c_char,
    message_cap: usize,
) -> i32;

/// Fallible callback for evaluating the gradient (NULL = no analytic
/// gradient). The status and message follow [`EindirTryEvalFn`].
pub type EindirTryGradFn = Option<
    unsafe extern "C" fn(
        user_data: *mut c_void,
        x: *const DLManagedTensorVersioned,
        grad_out: *mut DLManagedTensorVersioned,
        message_out: *mut c_char,
        message_cap: usize,
    ) -> i32,
>;

/// Embeddable, C-ABI-compatible objective handle with fallible callbacks.
///
/// Created by [`eindir_try_objective_new`] and released by
/// [`eindir_try_objective_free`]. A consumer that embeds it as the first
/// member of its own struct fills the fields itself and frees its own
/// struct, as with `eindir_objective_t`.
#[repr(C)]
pub struct eindir_try_objective_t {
    /// Number of input dimensions.
    pub dim: usize,
    /// Lower bounds: `dim` doubles, owned by this struct when created via
    /// `eindir_try_objective_new`.
    pub low: *mut f64,
    /// Upper bounds: `dim` doubles, owned by this struct when created via
    /// `eindir_try_objective_new`.
    pub high: *mut f64,
    /// Value callback.  Must not be NULL.
    pub try_eval_fn: EindirTryEvalFn,
    /// Gradient callback.  NULL when no analytic gradient is available.
    pub try_grad_fn: EindirTryGradFn,
    /// Opaque context forwarded verbatim to every callback invocation.
    pub user_data: *mut c_void,
    /// Optional destructor for `user_data`, called by
    /// `eindir_try_objective_free`.
    pub free_fn: Option<unsafe extern "C" fn(*mut c_void)>,
}

// Safety: user_data pointer is opaque; caller guarantees thread safety.
unsafe impl Send for eindir_try_objective_t {}
unsafe impl Sync for eindir_try_objective_t {}

/// A failure reported by a fallible C callback: its status and its message.
///
/// This is the `Error` type of [`EindirTryObjectiveWrapper`], so it reaches
/// a Rust driver without conversion.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
#[error("{}", self.describe())]
pub struct CallbackError {
    /// The nonzero status the callback returned.
    pub code: i32,
    /// The message the callback wrote, empty when it wrote none.
    pub message: String,
}

impl CallbackError {
    /// Builds an error from a status and a message.
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// Returns the error with `message` replaced.
    #[must_use]
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = message.into();
        self
    }

    fn describe(&self) -> String {
        if self.message.is_empty() {
            format!("callback failed with status {}", self.code)
        } else {
            format!(
                "callback failed with status {}: {}",
                self.code, self.message
            )
        }
    }
}

/// Run `f`, turning a panic into [`EINDIR_TRY_INTERNAL_ERROR`].
fn guard<F>(f: F) -> (i32, String)
where
    F: FnOnce() -> (i32, String) + std::panic::UnwindSafe,
{
    match std::panic::catch_unwind(f) {
        Ok(result) => result,
        Err(payload) => {
            let message = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            (EINDIR_TRY_INTERNAL_ERROR, message)
        }
    }
}

/// Copy `message` into the caller's buffer, truncated to fit with its NUL.
fn write_message(out: *mut c_char, cap: usize, message: &str) {
    if out.is_null() || cap == 0 {
        return;
    }
    let bytes = message.as_bytes();
    let n = bytes.len().min(cap - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr().cast::<c_char>(), out, n);
        *out.add(n) = 0;
    }
}

/// Read the NUL-terminated message a callback left in `buf`.
fn read_message(buf: &[c_char; EINDIR_TRY_MESSAGE_CAP]) -> String {
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Call the value callback and turn its status and message into a result.
unsafe fn call_eval(
    obj: &eindir_try_objective_t,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> Result<(), CallbackError> {
    let mut buf = [0 as c_char; EINDIR_TRY_MESSAGE_CAP];
    let status = unsafe {
        (obj.try_eval_fn)(
            obj.user_data,
            x,
            value_out,
            buf.as_mut_ptr(),
            EINDIR_TRY_MESSAGE_CAP,
        )
    };
    if status == EINDIR_TRY_SUCCESS {
        Ok(())
    } else {
        buf[EINDIR_TRY_MESSAGE_CAP - 1] = 0;
        Err(CallbackError::new(status, read_message(&buf)))
    }
}

/// Call the gradient callback, or fail when there is none.
unsafe fn call_grad(
    obj: &eindir_try_objective_t,
    x: *const DLManagedTensorVersioned,
    grad_out: *mut DLManagedTensorVersioned,
) -> Result<(), CallbackError> {
    let Some(grad_fn) = obj.try_grad_fn else {
        return Err(CallbackError::new(
            EINDIR_TRY_INVALID_PARAMETER,
            "no gradient callback",
        ));
    };
    let mut buf = [0 as c_char; EINDIR_TRY_MESSAGE_CAP];
    let status = unsafe {
        grad_fn(
            obj.user_data,
            x,
            grad_out,
            buf.as_mut_ptr(),
            EINDIR_TRY_MESSAGE_CAP,
        )
    };
    if status == EINDIR_TRY_SUCCESS {
        Ok(())
    } else {
        buf[EINDIR_TRY_MESSAGE_CAP - 1] = 0;
        Err(CallbackError::new(status, read_message(&buf)))
    }
}

/// Report a result to a C caller: the status is the return value, and the
/// message goes to the caller's buffer and to [`crate::ffi::eindir_last_error`].
fn report(result: (i32, String), message_out: *mut c_char, message_cap: usize) -> i32 {
    let (status, message) = result;
    if status == EINDIR_TRY_SUCCESS {
        write_message(message_out, message_cap, "");
        return status;
    }
    let shown = if message.is_empty() {
        format!("callback failed with status {status}")
    } else {
        message
    };
    set_last_error(&shown);
    write_message(message_out, message_cap, &shown);
    status
}

/// Create a new heap-allocated objective handle from fallible C callbacks.
///
/// Bounds data is copied from the DLPack tensors; the caller may free them
/// after this call returns.  Returns NULL on failure, with the reason in
/// `eindir_last_error()`; `free_fn` is not called then.
///
/// The caller must eventually pass the returned pointer to
/// [`eindir_try_objective_free`].
///
/// # Safety
///
/// Both bounds pointers must reference readable one-dimensional DLPack tensors
/// containing `dim` contiguous `f64` values. The callbacks and `user_data` must
/// remain valid until the returned handle is freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eindir_try_objective_new(
    dim: usize,
    bounds_low: *const DLManagedTensorVersioned,
    bounds_high: *const DLManagedTensorVersioned,
    try_eval_fn: EindirTryEvalFn,
    try_grad_fn: EindirTryGradFn,
    user_data: *mut c_void,
    free_fn: Option<unsafe extern "C" fn(*mut c_void)>,
) -> *mut eindir_try_objective_t {
    let caller = "eindir_try_objective_new";
    let Some(low) = (unsafe { copy_bounds(caller, bounds_low, "bounds_low", dim) }) else {
        return std::ptr::null_mut();
    };
    let Some(high) = (unsafe { copy_bounds(caller, bounds_high, "bounds_high", dim) }) else {
        return std::ptr::null_mut();
    };
    let low = Box::into_raw(low.into_boxed_slice()).cast::<f64>();
    let high = Box::into_raw(high.into_boxed_slice()).cast::<f64>();
    Box::into_raw(Box::new(eindir_try_objective_t {
        dim,
        low,
        high,
        try_eval_fn,
        try_grad_fn,
        user_data,
        free_fn,
    }))
}

/// Free a handle created by [`eindir_try_objective_new`].
///
/// Calls `free_fn(user_data)` if a destructor was registered, frees the bounds
/// arrays, then frees the struct.  Do NOT call this on embedded handles.
///
/// # Safety
///
/// `obj` must be null or a live handle returned by
/// [`eindir_try_objective_new`], and the handle must not be used or freed
/// again after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eindir_try_objective_free(obj: *mut eindir_try_objective_t) {
    if obj.is_null() {
        return;
    }
    let o = unsafe { &*obj };
    if let Some(free_fn) = o.free_fn
        && !o.user_data.is_null()
    {
        unsafe { free_fn(o.user_data) };
    }
    if !o.low.is_null() {
        unsafe {
            drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                o.low, o.dim,
            )))
        };
    }
    if !o.high.is_null() {
        unsafe {
            drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                o.high, o.dim,
            )))
        };
    }
    unsafe { drop(Box::from_raw(obj)) };
}

/// Returns the number of input dimensions, or 0 for NULL.
///
/// # Safety
///
/// `obj` must be null or point to a live [`eindir_try_objective_t`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eindir_try_objective_dim(obj: *const eindir_try_objective_t) -> usize {
    if obj.is_null() {
        return 0;
    }
    unsafe { (*obj).dim }
}

/// Returns non-zero if the handle has a gradient callback.
///
/// # Safety
///
/// `obj` must be null or point to a live [`eindir_try_objective_t`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eindir_try_objective_has_grad(obj: *const eindir_try_objective_t) -> i32 {
    if obj.is_null() {
        return 0;
    }
    i32::from(unsafe { (*obj).try_grad_fn.is_some() })
}

/// Evaluate the objective at point `x`.
///
/// Returns `0` on success, and otherwise the callback's own nonzero status.
/// A NULL argument gives [`EINDIR_TRY_INVALID_PARAMETER`] and a panic gives
/// [`EINDIR_TRY_INTERNAL_ERROR`]. On failure the message is written to
/// `message_out` (when it is not NULL and `message_cap` is not 0, truncated
/// to fit with its NUL) and also recorded for `eindir_last_error()`. On
/// success `message_out` is set to the empty string.
///
/// # Safety
///
/// `obj` must point to a live handle, `x` must be readable by its
/// registered callback, `value_out` must point to writable `f64` storage, and
/// `message_out` must be NULL or point to `message_cap` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eindir_try_objective_eval(
    obj: *const eindir_try_objective_t,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
    message_out: *mut c_char,
    message_cap: usize,
) -> i32 {
    let result = guard(std::panic::AssertUnwindSafe(|| {
        if obj.is_null() || x.is_null() || value_out.is_null() {
            return (
                EINDIR_TRY_INVALID_PARAMETER,
                "eindir_try_objective_eval: NULL argument".to_string(),
            );
        }
        match unsafe { call_eval(&*obj, x, value_out) } {
            Ok(()) => (EINDIR_TRY_SUCCESS, String::new()),
            Err(error) => (error.code, error.message),
        }
    }));
    report(result, message_out, message_cap)
}

/// Compute the gradient at point `x`.
///
/// Returns `0` on success, and otherwise the callback's own nonzero status.
/// A handle without a gradient callback gives
/// [`EINDIR_TRY_INVALID_PARAMETER`] with "no gradient callback". The message
/// follows [`eindir_try_objective_eval`].
///
/// # Safety
///
/// `obj` must point to a live handle; `x` and `grad_out` must satisfy the
/// registered gradient callback's readable-input and writable-output
/// contracts, and `message_out` must be NULL or point to `message_cap`
/// writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eindir_try_objective_grad(
    obj: *const eindir_try_objective_t,
    x: *const DLManagedTensorVersioned,
    grad_out: *mut DLManagedTensorVersioned,
    message_out: *mut c_char,
    message_cap: usize,
) -> i32 {
    let result = guard(std::panic::AssertUnwindSafe(|| {
        if obj.is_null() || x.is_null() || grad_out.is_null() {
            return (
                EINDIR_TRY_INVALID_PARAMETER,
                "eindir_try_objective_grad: NULL argument".to_string(),
            );
        }
        match unsafe { call_grad(&*obj, x, grad_out) } {
            Ok(()) => (EINDIR_TRY_SUCCESS, String::new()),
            Err(error) => (error.code, error.message),
        }
    }));
    report(result, message_out, message_cap)
}

/// Rust-side view over a fallible C objective.
///
/// Its `Error` is [`CallbackError`], so a driver generic over
/// [`TryObjective`] returns the callback's status and message unchanged.
pub struct EindirTryObjectiveWrapper<'a> {
    obj: &'a eindir_try_objective_t,
    bounds: Bounds<f64>,
}

impl<'a> EindirTryObjectiveWrapper<'a> {
    /// Wraps a handle.
    ///
    /// # Safety
    /// `obj.low` and `obj.high` must point to `obj.dim` valid f64 values.
    pub unsafe fn new(obj: &'a eindir_try_objective_t) -> Self {
        let low = unsafe { std::slice::from_raw_parts(obj.low, obj.dim) };
        let high = unsafe { std::slice::from_raw_parts(obj.high, obj.dim) };
        Self {
            obj,
            bounds: Bounds::new(
                Array1::from_vec(low.to_vec()),
                Array1::from_vec(high.to_vec()),
                0.0,
            ),
        }
    }
}

impl TryObjective<f64> for EindirTryObjectiveWrapper<'_> {
    type Error = CallbackError;

    fn dims(&self) -> usize {
        self.obj.dim
    }

    fn domain(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn try_eval(&self, x: ArrayView1<f64>) -> Result<f64, CallbackError> {
        let mut data = x.to_owned();
        let len = data.len();
        let tensor = unsafe { create_borrowed_f64_1d(data.as_mut_ptr(), len) };
        let mut value = f64::NAN;
        let outcome = unsafe { call_eval(self.obj, tensor, &mut value) };
        unsafe { tensor_free(tensor) };
        outcome.map(|()| value)
    }
}

impl TryGradient<f64> for EindirTryObjectiveWrapper<'_> {
    type GradError = CallbackError;

    fn try_grad(&self, x: ArrayView1<f64>) -> Result<Array1<f64>, CallbackError> {
        let mut x_data = x.to_owned();
        let len = x_data.len();
        let x_tensor = unsafe { create_borrowed_f64_1d(x_data.as_mut_ptr(), len) };
        let mut grad_data = vec![0.0_f64; self.obj.dim];
        let grad_tensor = unsafe { create_borrowed_f64_1d(grad_data.as_mut_ptr(), self.obj.dim) };
        let outcome = unsafe { call_grad(self.obj, x_tensor, grad_tensor) };
        unsafe {
            tensor_free(x_tensor);
            tensor_free(grad_tensor);
        }
        outcome.map(|()| Array1::from_vec(grad_data))
    }

    fn grad_dims(&self) -> usize {
        self.obj.dim
    }
}

impl TryDifferentiableObjective<f64> for EindirTryObjectiveWrapper<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{EINDIR_ABI_FEATURE_TRY_CALLBACKS, eindir_core_abi_stamp};
    use ndarray::array;
    use std::ffi::CStr;

    fn last_error_text() -> String {
        let ptr = crate::ffi::eindir_last_error();
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }

    const REFUSED: i32 = 7;
    const BROKEN: i32 = 9;

    fn put_message(out: *mut c_char, cap: usize, text: &str) {
        write_message(out, cap, text);
    }

    unsafe fn read_vec<'a>(x: *const DLManagedTensorVersioned) -> &'a [f64] {
        let t = unsafe { &(*x).dl_tensor };
        let n = unsafe { *t.shape } as usize;
        unsafe { std::slice::from_raw_parts(t.data as *const f64, n) }
    }

    /// Sum of squares; refuses a first coordinate above 10 with status 7
    /// and a message, and fails with status 9 and no message on NaN.
    unsafe extern "C" fn sphere_eval(
        _ud: *mut c_void,
        x: *const DLManagedTensorVersioned,
        out: *mut f64,
        message_out: *mut c_char,
        message_cap: usize,
    ) -> i32 {
        let xs = unsafe { read_vec(x) };
        if xs.iter().any(|v| v.is_nan()) {
            return BROKEN;
        }
        if xs[0] > 10.0 {
            put_message(message_out, message_cap, "geometry refused: atom 0 too far");
            return REFUSED;
        }
        unsafe { *out = xs.iter().map(|v| v * v).sum() };
        EINDIR_TRY_SUCCESS
    }

    unsafe extern "C" fn sphere_grad(
        _ud: *mut c_void,
        x: *const DLManagedTensorVersioned,
        grad_out: *mut DLManagedTensorVersioned,
        message_out: *mut c_char,
        message_cap: usize,
    ) -> i32 {
        let xs = unsafe { read_vec(x) };
        if xs[0] > 10.0 {
            put_message(message_out, message_cap, "forces refused");
            return REFUSED;
        }
        let t = unsafe { &(*grad_out).dl_tensor };
        let g = unsafe { std::slice::from_raw_parts_mut(t.data as *mut f64, xs.len()) };
        for (gi, xi) in g.iter_mut().zip(xs) {
            *gi = 2.0 * xi;
        }
        EINDIR_TRY_SUCCESS
    }

    unsafe extern "C" fn long_message_eval(
        _ud: *mut c_void,
        _x: *const DLManagedTensorVersioned,
        _out: *mut f64,
        message_out: *mut c_char,
        message_cap: usize,
    ) -> i32 {
        put_message(message_out, message_cap, &"m".repeat(2 * message_cap));
        REFUSED
    }

    fn make(
        eval: EindirTryEvalFn,
        grad: EindirTryGradFn,
        dim: usize,
    ) -> *mut eindir_try_objective_t {
        let mut low = vec![-20.0_f64; dim];
        let mut high = vec![20.0_f64; dim];
        let low_t = unsafe { create_borrowed_f64_1d(low.as_mut_ptr(), dim) };
        let high_t = unsafe { create_borrowed_f64_1d(high.as_mut_ptr(), dim) };
        let obj = unsafe {
            eindir_try_objective_new(dim, low_t, high_t, eval, grad, std::ptr::null_mut(), None)
        };
        unsafe {
            tensor_free(low_t);
            tensor_free(high_t);
        }
        obj
    }

    fn message_text(buf: &[c_char; 64]) -> String {
        let bytes: Vec<u8> = buf
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect();
        String::from_utf8(bytes).expect("ASCII message")
    }

    #[test]
    fn c_eval_returns_the_callback_status_and_message() {
        let obj = make(sphere_eval, Some(sphere_grad), 2);
        assert!(!obj.is_null());
        assert_eq!(unsafe { eindir_try_objective_dim(obj) }, 2);
        assert_eq!(unsafe { eindir_try_objective_has_grad(obj) }, 1);
        let mut buf = [0 as c_char; 64];

        let mut ok = [1.0, 2.0];
        let ok_t = unsafe { create_borrowed_f64_1d(ok.as_mut_ptr(), 2) };
        let mut value = 0.0;
        buf[0] = b'x' as c_char;
        let status =
            unsafe { eindir_try_objective_eval(obj, ok_t, &mut value, buf.as_mut_ptr(), 64) };
        assert_eq!(status, EINDIR_TRY_SUCCESS);
        assert_eq!(value, 5.0);
        assert_eq!(message_text(&buf), "", "success clears the buffer");

        let mut far = [11.0, 0.0];
        let far_t = unsafe { create_borrowed_f64_1d(far.as_mut_ptr(), 2) };
        let status =
            unsafe { eindir_try_objective_eval(obj, far_t, &mut value, buf.as_mut_ptr(), 64) };
        assert_eq!(status, REFUSED);
        assert_eq!(message_text(&buf), "geometry refused: atom 0 too far");
        assert_eq!(last_error_text(), "geometry refused: atom 0 too far");

        let mut nan = [f64::NAN, 0.0];
        let nan_t = unsafe { create_borrowed_f64_1d(nan.as_mut_ptr(), 2) };
        let status =
            unsafe { eindir_try_objective_eval(obj, nan_t, &mut value, std::ptr::null_mut(), 0) };
        assert_eq!(status, BROKEN, "a NULL message buffer is allowed");
        assert_eq!(last_error_text(), "callback failed with status 9");

        unsafe {
            tensor_free(ok_t);
            tensor_free(far_t);
            tensor_free(nan_t);
            eindir_try_objective_free(obj);
        }
    }

    #[test]
    fn c_grad_returns_the_callback_status_and_message() {
        let obj = make(sphere_eval, Some(sphere_grad), 2);
        let mut buf = [0 as c_char; 64];
        let mut x = [1.0, 2.0];
        let x_t = unsafe { create_borrowed_f64_1d(x.as_mut_ptr(), 2) };
        let mut g = [0.0; 2];
        let g_t = unsafe { create_borrowed_f64_1d(g.as_mut_ptr(), 2) };
        let status = unsafe { eindir_try_objective_grad(obj, x_t, g_t, buf.as_mut_ptr(), 64) };
        assert_eq!(status, EINDIR_TRY_SUCCESS);
        assert_eq!(g, [2.0, 4.0]);

        let mut far = [11.0, 0.0];
        let far_t = unsafe { create_borrowed_f64_1d(far.as_mut_ptr(), 2) };
        let status = unsafe { eindir_try_objective_grad(obj, far_t, g_t, buf.as_mut_ptr(), 64) };
        assert_eq!(status, REFUSED);
        assert_eq!(message_text(&buf), "forces refused");
        unsafe {
            tensor_free(x_t);
            tensor_free(g_t);
            tensor_free(far_t);
            eindir_try_objective_free(obj);
        }
    }

    #[test]
    fn c_entry_points_report_misuse_and_missing_gradient() {
        let obj = make(sphere_eval, None, 2);
        assert_eq!(unsafe { eindir_try_objective_has_grad(obj) }, 0);
        let mut x = [1.0, 2.0];
        let x_t = unsafe { create_borrowed_f64_1d(x.as_mut_ptr(), 2) };
        let mut g = [0.0; 2];
        let g_t = unsafe { create_borrowed_f64_1d(g.as_mut_ptr(), 2) };
        let mut buf = [0 as c_char; 64];
        let status = unsafe { eindir_try_objective_grad(obj, x_t, g_t, buf.as_mut_ptr(), 64) };
        assert_eq!(status, EINDIR_TRY_INVALID_PARAMETER);
        assert_eq!(message_text(&buf), "no gradient callback");

        let mut value = 0.0;
        let status = unsafe {
            eindir_try_objective_eval(std::ptr::null(), x_t, &mut value, buf.as_mut_ptr(), 64)
        };
        assert_eq!(status, EINDIR_TRY_INVALID_PARAMETER);
        assert_eq!(
            message_text(&buf),
            "eindir_try_objective_eval: NULL argument"
        );
        assert_eq!(unsafe { eindir_try_objective_dim(std::ptr::null()) }, 0);
        assert_eq!(
            unsafe { eindir_try_objective_has_grad(std::ptr::null()) },
            0
        );
        unsafe {
            tensor_free(x_t);
            tensor_free(g_t);
            eindir_try_objective_free(obj);
            eindir_try_objective_free(std::ptr::null_mut());
        }
    }

    #[test]
    fn a_long_message_is_truncated_to_the_buffer() {
        let obj = make(long_message_eval, None, 1);
        let mut x = [0.0];
        let x_t = unsafe { create_borrowed_f64_1d(x.as_mut_ptr(), 1) };
        let mut value = 0.0;
        let mut buf = [0x7f as c_char; 8];
        let status =
            unsafe { eindir_try_objective_eval(obj, x_t, &mut value, buf.as_mut_ptr(), 8) };
        assert_eq!(status, REFUSED);
        let mut expected = [b'm' as c_char; 8];
        expected[7] = 0;
        assert_eq!(buf, expected);
        unsafe {
            tensor_free(x_t);
            eindir_try_objective_free(obj);
        }
    }

    #[test]
    fn a_panic_behind_the_boundary_is_an_internal_error() {
        let result = guard(|| panic!("boom"));
        assert_eq!(result, (EINDIR_TRY_INTERNAL_ERROR, "boom".to_string()));
        let result = guard(|| std::panic::panic_any(5_u8));
        assert_eq!(
            result,
            (EINDIR_TRY_INTERNAL_ERROR, "unknown panic".to_string())
        );
    }

    #[test]
    fn creation_refuses_bad_bounds() {
        let mut low = vec![0.0_f64; 2];
        let low_t = unsafe { create_borrowed_f64_1d(low.as_mut_ptr(), 2) };
        let obj = unsafe {
            eindir_try_objective_new(
                3,
                low_t,
                low_t,
                sphere_eval,
                None,
                std::ptr::null_mut(),
                None,
            )
        };
        assert!(obj.is_null());
        assert_eq!(
            last_error_text(),
            "eindir_try_objective_new: bounds_low length 2 != dim 3"
        );
        unsafe { tensor_free(low_t) };
    }

    #[test]
    fn the_rust_wrapper_passes_the_status_and_message_through() {
        let obj = make(sphere_eval, Some(sphere_grad), 2);
        let wrapper = unsafe { EindirTryObjectiveWrapper::new(&*obj) };
        assert_eq!(wrapper.dims(), 2);
        assert_eq!(wrapper.try_eval(array![1.0, 2.0].view()), Ok(5.0));
        assert_eq!(
            wrapper.try_grad(array![1.0, 2.0].view()),
            Ok(array![2.0, 4.0])
        );
        assert_eq!(
            wrapper.try_value_and_gradient(array![1.0, 1.0].view()),
            Ok((2.0, array![2.0, 2.0]))
        );
        let refused = CallbackError::new(REFUSED, "geometry refused: atom 0 too far");
        assert_eq!(wrapper.try_eval(array![11.0, 0.0].view()), Err(refused));
        assert_eq!(
            wrapper.try_grad(array![11.0, 0.0].view()),
            Err(CallbackError::new(REFUSED, "forces refused"))
        );
        assert_eq!(
            wrapper.try_eval(array![f64::NAN, 0.0].view()),
            Err(CallbackError::new(BROKEN, ""))
        );
        let batch = array![[1.0, 0.0], [12.0, 0.0], [2.0, 0.0]];
        assert_eq!(
            wrapper.try_eval_batch(batch.view()).map_err(|e| e.code),
            Err(REFUSED)
        );
        unsafe { eindir_try_objective_free(obj) };
    }

    #[test]
    fn the_rust_wrapper_without_a_gradient_callback_fails_instead_of_zeroing() {
        let obj = make(sphere_eval, None, 2);
        let wrapper = unsafe { EindirTryObjectiveWrapper::new(&*obj) };
        assert_eq!(
            wrapper.try_grad(array![1.0, 2.0].view()),
            Err(CallbackError::new(
                EINDIR_TRY_INVALID_PARAMETER,
                "no gradient callback"
            ))
        );
        unsafe { eindir_try_objective_free(obj) };
    }

    #[test]
    fn callback_error_displays_status_and_message() {
        let with = CallbackError::new(7, "refused");
        assert_eq!(with.to_string(), "callback failed with status 7: refused");
        let without = CallbackError::new(9, "");
        assert_eq!(without.to_string(), "callback failed with status 9");
        let renamed = without.with_message("late");
        assert_eq!(renamed.message, "late");
    }

    #[test]
    fn the_stamp_advertises_the_fallible_callbacks() {
        let stamp = eindir_core_abi_stamp();
        assert!(stamp.features & EINDIR_ABI_FEATURE_TRY_CALLBACKS != 0);
        assert_eq!(stamp.objective_layout, 3, "the old handle is unchanged");
    }

    #[test]
    fn a_free_fn_runs_once_with_the_user_data() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static FREED: AtomicUsize = AtomicUsize::new(0);
        unsafe extern "C" fn free_counter(ud: *mut c_void) {
            assert!(!ud.is_null());
            FREED.fetch_add(1, Ordering::SeqCst);
        }
        let mut low = vec![-1.0_f64];
        let mut high = vec![1.0_f64];
        let low_t = unsafe { create_borrowed_f64_1d(low.as_mut_ptr(), 1) };
        let high_t = unsafe { create_borrowed_f64_1d(high.as_mut_ptr(), 1) };
        let mut context = 0_u8;
        let obj = unsafe {
            eindir_try_objective_new(
                1,
                low_t,
                high_t,
                sphere_eval,
                None,
                (&mut context as *mut u8).cast(),
                Some(free_counter),
            )
        };
        unsafe {
            tensor_free(low_t);
            tensor_free(high_t);
            eindir_try_objective_free(obj);
        }
        assert_eq!(FREED.load(Ordering::SeqCst), 1);
    }
}
