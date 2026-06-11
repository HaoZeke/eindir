//! C ABI surface for eindir-core.
//!
//! ## Version
//! [`eindir_core_version`] returns the package version as a NUL-terminated
//! ASCII string.
//!
//! ## Objective handle
//! A C or C++ caller can define an objective from its own value (and optional
//! gradient) function pointers and evaluate it through eindir.  Arrays are
//! passed as DLPack `DLManagedTensorVersioned*` tensors so the data can live
//! on any device without eindir itself taking on a CUDA dependency.
//!
//! ### Lifecycle
//! ```c
//! eindir_objective_t *obj = eindir_objective_new(
//!     dim, low_tensor, high_tensor, eval_fn, grad_fn, user_data, free_fn);
//! double val = 0.0;
//! eindir_status_t s = eindir_objective_eval(obj, x_tensor, &val);
//! eindir_objective_free(obj);
//! ```

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::{c_char, c_void};

use dlpk::sys::DLManagedTensorVersioned;

// ---------------------------------------------------------------------------
// Version
// ---------------------------------------------------------------------------

/// Returns the eindir-core package version as a NUL-terminated ASCII string.
///
/// The returned pointer is valid for the lifetime of the process — the string
/// lives in the binary's read-only data segment and is never freed.
#[unsafe(no_mangle)]
pub extern "C" fn eindir_core_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

// ---------------------------------------------------------------------------
// Status codes
// ---------------------------------------------------------------------------

/// Status codes returned by all eindir C API functions.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum eindir_status_t {
    /// Operation completed successfully.
    EINDIR_SUCCESS = 0,
    /// An invalid parameter was passed (null pointer, wrong size, etc.).
    EINDIR_INVALID_PARAMETER = 1,
    /// An internal error occurred (e.g. a Rust panic was caught).
    EINDIR_INTERNAL_ERROR = 2,
}

// ---------------------------------------------------------------------------
// Thread-local error message
// ---------------------------------------------------------------------------

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_last_error(msg: &str) {
    LAST_ERROR.with(|cell| {
        let c = CString::new(msg).unwrap_or_else(|_| {
            CString::new("(error message contained interior NUL)").unwrap()
        });
        *cell.borrow_mut() = c;
    });
}

/// Retrieve the last error message for the current thread.
///
/// The pointer is valid until the next `eindir_*` call on the same thread.
#[unsafe(no_mangle)]
pub extern "C" fn eindir_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| cell.borrow().as_ptr())
}

fn catch_unwind<F>(f: F) -> eindir_status_t
where
    F: FnOnce() -> eindir_status_t + std::panic::UnwindSafe,
{
    match std::panic::catch_unwind(f) {
        Ok(status) => status,
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            set_last_error(&msg);
            eindir_status_t::EINDIR_INTERNAL_ERROR
        }
    }
}

// ---------------------------------------------------------------------------
// Callback types
// ---------------------------------------------------------------------------

/// Callback for evaluating the objective value.
///
/// - `user_data`: opaque pointer forwarded from `eindir_objective_new`.
/// - `x`: a DLPack tensor of shape `[dim]`, dtype f64, on any device.
/// - `value_out`: pointer where the callback writes the scalar result.
///
/// Returns `EINDIR_SUCCESS` on success, or an error status code.
pub type EindirEvalFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> eindir_status_t;

/// Callback for evaluating the gradient.
///
/// - `user_data`: opaque pointer forwarded from `eindir_objective_new`.
/// - `x`: a DLPack tensor of shape `[dim]`, dtype f64.
/// - `grad_out`: a DLPack tensor of shape `[dim]`, dtype f64 — the callback
///   writes the gradient values into this tensor's data buffer.
///
/// Returns `EINDIR_SUCCESS` on success.
pub type EindirGradFn = Option<
    unsafe extern "C" fn(
        user_data: *mut c_void,
        x: *const DLManagedTensorVersioned,
        grad_out: *mut DLManagedTensorVersioned,
    ) -> eindir_status_t,
>;

/// Destructor for `user_data`.
pub type EindirFreeFn = Option<unsafe extern "C" fn(*mut c_void)>;

// ---------------------------------------------------------------------------
// Opaque objective handle
// ---------------------------------------------------------------------------

/// Opaque handle representing a user-defined objective function.
///
/// Created via [`eindir_objective_new`], freed via [`eindir_objective_free`].
pub struct eindir_objective_t {
    dim: usize,
    /// Bounds lower corner, owned.
    low: Vec<f64>,
    /// Bounds upper corner, owned.
    high: Vec<f64>,
    /// Cached `Bounds<f64>` for the `Objective` trait impl (lazily initialized).
    bounds_cache: std::sync::OnceLock<crate::Bounds<f64>>,
    eval_fn: EindirEvalFn,
    grad_fn: EindirGradFn,
    user_data: *mut c_void,
    free_fn: EindirFreeFn,
}

unsafe impl Send for eindir_objective_t {}
// Safety: The user_data pointer is opaque and the caller guarantees thread
// safety of the underlying object, matching the contract in rgpot/metatensor.
unsafe impl Sync for eindir_objective_t {}

impl Drop for eindir_objective_t {
    fn drop(&mut self) {
        if let Some(free_fn) = self.free_fn {
            if !self.user_data.is_null() {
                unsafe { free_fn(self.user_data) };
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DLPack tensor helpers (internal, for bounds tensors)
// ---------------------------------------------------------------------------

/// Create a read-only, non-owning 1-D f64 DLPack tensor wrapping `data`.
///
/// # Safety
/// `data` must point to at least `len` contiguous f64 values and must
/// remain valid for the lifetime of the returned tensor.
unsafe fn create_borrowed_f64_1d(
    data: *mut f64,
    len: usize,
) -> *mut DLManagedTensorVersioned {
    use dlpk::sys::{
        DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLPackVersion, DLTensor,
    };

    struct Ctx {
        shape: [i64; 1],
        strides: [i64; 1],
    }

    unsafe extern "C" fn deleter(ptr: *mut DLManagedTensorVersioned) {
        if ptr.is_null() {
            return;
        }
        let ctx = unsafe { (*ptr).manager_ctx.cast::<Ctx>() };
        if !ctx.is_null() {
            drop(unsafe { Box::from_raw(ctx) });
        }
        drop(unsafe { Box::from_raw(ptr) });
    }

    let mut ctx = Box::new(Ctx {
        shape: [len as i64],
        strides: [1],
    });

    let dl_tensor = DLTensor {
        data: data.cast(),
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
        shape: ctx.shape.as_mut_ptr(),
        strides: ctx.strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let managed = Box::new(DLManagedTensorVersioned {
        version: DLPackVersion { major: 1, minor: 0 },
        manager_ctx: Box::into_raw(ctx).cast(),
        deleter: Some(deleter),
        flags: 0,
        dl_tensor,
    });

    Box::into_raw(managed)
}

/// Free a DLPack tensor by invoking its deleter.
unsafe fn tensor_free(tensor: *mut DLManagedTensorVersioned) {
    if tensor.is_null() {
        return;
    }
    if let Some(deleter) = unsafe { (*tensor).deleter } {
        unsafe { deleter(tensor) };
    }
}

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

/// Create a new objective handle from C callbacks.
///
/// - `dim`: number of input dimensions.
/// - `bounds_low`: DLPack tensor of shape `[dim]`, dtype f64 — lower bounds.
///   The data is **copied**; the caller may free the tensor after this call.
/// - `bounds_high`: DLPack tensor of shape `[dim]`, dtype f64 — upper bounds.
///   The data is **copied**.
/// - `eval_fn`: callback that computes the objective value (required).
/// - `grad_fn`: callback that computes the gradient (may be `NULL`).
/// - `user_data`: opaque pointer forwarded to the callbacks.
/// - `free_fn`: optional destructor for `user_data` (may be `NULL`).
///
/// Returns a heap-allocated `eindir_objective_t*`, or `NULL` on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eindir_objective_new(
    dim: usize,
    bounds_low: *const DLManagedTensorVersioned,
    bounds_high: *const DLManagedTensorVersioned,
    eval_fn: EindirEvalFn,
    grad_fn: EindirGradFn,
    user_data: *mut c_void,
    free_fn: EindirFreeFn,
) -> *mut eindir_objective_t {
    // Validate and copy bounds
    let copy_bounds = |tensor: *const DLManagedTensorVersioned, name: &str| -> Option<Vec<f64>> {
        if tensor.is_null() {
            set_last_error(&format!("eindir_objective_new: {name} is NULL"));
            return None;
        }
        let t = unsafe { &(*tensor).dl_tensor };
        if t.ndim != 1 {
            set_last_error(&format!(
                "eindir_objective_new: {name} must be 1-D, got ndim={}",
                t.ndim
            ));
            return None;
        }
        let len = unsafe { *t.shape } as usize;
        if len != dim {
            set_last_error(&format!(
                "eindir_objective_new: {name} length {len} != dim {dim}"
            ));
            return None;
        }
        let data = t.data as *const f64;
        if data.is_null() {
            set_last_error(&format!("eindir_objective_new: {name} data is NULL"));
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts(data, len) }.to_vec())
    };

    let low = match copy_bounds(bounds_low, "bounds_low") {
        Some(v) => v,
        None => return std::ptr::null_mut(),
    };
    let high = match copy_bounds(bounds_high, "bounds_high") {
        Some(v) => v,
        None => return std::ptr::null_mut(),
    };

    let obj = eindir_objective_t {
        dim,
        low,
        high,
        bounds_cache: std::sync::OnceLock::new(),
        eval_fn,
        grad_fn,
        user_data,
        free_fn,
    };
    Box::into_raw(Box::new(obj))
}

/// Free an objective handle.
///
/// If `obj` is `NULL`, this is a no-op.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eindir_objective_free(obj: *mut eindir_objective_t) {
    if !obj.is_null() {
        drop(unsafe { Box::from_raw(obj) });
    }
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

/// Returns the number of input dimensions.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eindir_objective_dim(
    obj: *const eindir_objective_t,
) -> usize {
    if obj.is_null() {
        return 0;
    }
    unsafe { (*obj).dim }
}

/// Writes the lower-bound vector into `out`, a DLPack tensor of shape `[dim]`.
///
/// The caller must supply a pre-allocated tensor whose data buffer has room
/// for `dim` f64 values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eindir_objective_bounds_low(
    obj: *const eindir_objective_t,
    out: *mut DLManagedTensorVersioned,
) -> eindir_status_t {
    catch_unwind(std::panic::AssertUnwindSafe(|| {
        if obj.is_null() || out.is_null() {
            set_last_error("eindir_objective_bounds_low: NULL argument");
            return eindir_status_t::EINDIR_INVALID_PARAMETER;
        }
        let o = unsafe { &*obj };
        let t = unsafe { &(*out).dl_tensor };
        let dst = t.data as *mut f64;
        if dst.is_null() {
            set_last_error("eindir_objective_bounds_low: tensor data is NULL");
            return eindir_status_t::EINDIR_INVALID_PARAMETER;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(o.low.as_ptr(), dst, o.dim);
        }
        eindir_status_t::EINDIR_SUCCESS
    }))
}

/// Writes the upper-bound vector into `out`, a DLPack tensor of shape `[dim]`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eindir_objective_bounds_high(
    obj: *const eindir_objective_t,
    out: *mut DLManagedTensorVersioned,
) -> eindir_status_t {
    catch_unwind(std::panic::AssertUnwindSafe(|| {
        if obj.is_null() || out.is_null() {
            set_last_error("eindir_objective_bounds_high: NULL argument");
            return eindir_status_t::EINDIR_INVALID_PARAMETER;
        }
        let o = unsafe { &*obj };
        let t = unsafe { &(*out).dl_tensor };
        let dst = t.data as *mut f64;
        if dst.is_null() {
            set_last_error("eindir_objective_bounds_high: tensor data is NULL");
            return eindir_status_t::EINDIR_INVALID_PARAMETER;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(o.high.as_ptr(), dst, o.dim);
        }
        eindir_status_t::EINDIR_SUCCESS
    }))
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// Evaluate the objective at point `x`.
///
/// - `obj`: a valid objective handle.
/// - `x`: DLPack tensor of shape `[dim]`, dtype f64.
/// - `value_out`: pointer where the result is written.
///
/// Returns `EINDIR_SUCCESS` on success.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eindir_objective_eval(
    obj: *const eindir_objective_t,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> eindir_status_t {
    catch_unwind(std::panic::AssertUnwindSafe(|| {
        if obj.is_null() {
            set_last_error("eindir_objective_eval: obj is NULL");
            return eindir_status_t::EINDIR_INVALID_PARAMETER;
        }
        if x.is_null() {
            set_last_error("eindir_objective_eval: x is NULL");
            return eindir_status_t::EINDIR_INVALID_PARAMETER;
        }
        if value_out.is_null() {
            set_last_error("eindir_objective_eval: value_out is NULL");
            return eindir_status_t::EINDIR_INVALID_PARAMETER;
        }
        let o = unsafe { &*obj };
        unsafe { (o.eval_fn)(o.user_data, x, value_out) }
    }))
}

/// Compute the gradient at point `x`.
///
/// - `obj`: a valid objective handle that was created with a non-NULL `grad_fn`.
/// - `x`: DLPack tensor of shape `[dim]`, dtype f64.
/// - `grad_out`: DLPack tensor of shape `[dim]`, dtype f64 — the gradient is
///   written into this tensor's data buffer.
///
/// Returns `EINDIR_INVALID_PARAMETER` if the objective has no gradient callback.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eindir_objective_grad(
    obj: *const eindir_objective_t,
    x: *const DLManagedTensorVersioned,
    grad_out: *mut DLManagedTensorVersioned,
) -> eindir_status_t {
    catch_unwind(std::panic::AssertUnwindSafe(|| {
        if obj.is_null() {
            set_last_error("eindir_objective_grad: obj is NULL");
            return eindir_status_t::EINDIR_INVALID_PARAMETER;
        }
        if x.is_null() {
            set_last_error("eindir_objective_grad: x is NULL");
            return eindir_status_t::EINDIR_INVALID_PARAMETER;
        }
        if grad_out.is_null() {
            set_last_error("eindir_objective_grad: grad_out is NULL");
            return eindir_status_t::EINDIR_INVALID_PARAMETER;
        }
        let o = unsafe { &*obj };
        match o.grad_fn {
            Some(gf) => unsafe { gf(o.user_data, x, grad_out) },
            None => {
                set_last_error("eindir_objective_grad: no gradient callback");
                eindir_status_t::EINDIR_INVALID_PARAMETER
            }
        }
    }))
}

/// Returns `true` (non-zero) if the objective has a gradient callback.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eindir_objective_has_grad(
    obj: *const eindir_objective_t,
) -> i32 {
    if obj.is_null() {
        return 0;
    }
    if unsafe { (*obj).grad_fn.is_some() } {
        1
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Rust-side trait implementation: allows consuming an eindir_objective_t
// as an Objective<f64> (and optionally Gradient<f64>) from Rust code.
// This is the bridge that lets rgpot-core potentials be used as eindir
// objectives on the Rust side too.
// ---------------------------------------------------------------------------

use crate::{Bounds, Objective};
use ndarray::{Array1, ArrayView1};

impl Objective<f64> for eindir_objective_t {
    fn dim(&self) -> usize {
        self.dim
    }

    fn bounds(&self) -> &Bounds<f64> {
        self.bounds_cache.get_or_init(|| {
            Bounds::new(
                Array1::from_vec(self.low.clone()),
                Array1::from_vec(self.high.clone()),
                0.0,
            )
        })
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        // Create a temporary borrowed DLPack tensor wrapping the ndarray data.
        let mut data = x.to_owned();
        let len = data.len();
        let tensor = unsafe { create_borrowed_f64_1d(data.as_mut_ptr(), len) };
        let mut value: f64 = f64::NAN;
        let status = unsafe { (self.eval_fn)(self.user_data, tensor, &mut value) };
        unsafe { tensor_free(tensor) };
        if status != eindir_status_t::EINDIR_SUCCESS {
            return f64::NAN;
        }
        value
    }
}

use crate::gradient::Gradient;

impl Gradient<f64> for eindir_objective_t {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let grad_fn = match self.grad_fn {
            Some(gf) => gf,
            None => return Array1::zeros(self.dim),
        };
        let mut x_data = x.to_owned();
        let len = x_data.len();
        let x_tensor = unsafe { create_borrowed_f64_1d(x_data.as_mut_ptr(), len) };

        let mut grad_data = vec![0.0f64; self.dim];
        let grad_tensor = unsafe { create_borrowed_f64_1d(grad_data.as_mut_ptr(), self.dim) };

        let _status = unsafe { grad_fn(self.user_data, x_tensor, grad_tensor) };
        unsafe {
            tensor_free(x_tensor);
            tensor_free(grad_tensor);
        }
        Array1::from_vec(grad_data)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_not_empty() {
        let ptr = eindir_core_version();
        let s = unsafe { std::ffi::CStr::from_ptr(ptr) };
        assert!(!s.to_str().unwrap().is_empty());
    }

    // A trivial eval callback: f(x) = sum(x)
    unsafe extern "C" fn sum_eval(
        _ud: *mut c_void,
        x: *const DLManagedTensorVersioned,
        out: *mut f64,
    ) -> eindir_status_t {
        let t = unsafe { &(*x).dl_tensor };
        let n = unsafe { *t.shape } as usize;
        let data = unsafe { std::slice::from_raw_parts(t.data as *const f64, n) };
        unsafe { *out = data.iter().sum() };
        eindir_status_t::EINDIR_SUCCESS
    }

    // A trivial grad callback: grad(x) = [1, 1, ..., 1]
    unsafe extern "C" fn ones_grad(
        _ud: *mut c_void,
        _x: *const DLManagedTensorVersioned,
        grad_out: *mut DLManagedTensorVersioned,
    ) -> eindir_status_t {
        let t = unsafe { &(*grad_out).dl_tensor };
        let n = unsafe { *t.shape } as usize;
        let data = unsafe { std::slice::from_raw_parts_mut(t.data as *mut f64, n) };
        for v in data.iter_mut() {
            *v = 1.0;
        }
        eindir_status_t::EINDIR_SUCCESS
    }

    fn make_test_objective(dim: usize) -> *mut eindir_objective_t {
        let mut low = vec![-5.0f64; dim];
        let mut high = vec![5.0f64; dim];
        let low_t = unsafe { create_borrowed_f64_1d(low.as_mut_ptr(), dim) };
        let high_t = unsafe { create_borrowed_f64_1d(high.as_mut_ptr(), dim) };

        let obj = unsafe {
            eindir_objective_new(
                dim,
                low_t,
                high_t,
                sum_eval,
                Some(ones_grad),
                std::ptr::null_mut(),
                None,
            )
        };
        unsafe {
            tensor_free(low_t);
            tensor_free(high_t);
        }
        obj
    }

    #[test]
    fn test_objective_lifecycle() {
        let obj = make_test_objective(3);
        assert!(!obj.is_null());
        assert_eq!(unsafe { eindir_objective_dim(obj) }, 3);
        assert_eq!(unsafe { eindir_objective_has_grad(obj) }, 1);
        unsafe { eindir_objective_free(obj) };
    }

    #[test]
    fn test_objective_eval() {
        let obj = make_test_objective(3);
        let mut x_data = [1.0, 2.0, 3.0];
        let x_t = unsafe { create_borrowed_f64_1d(x_data.as_mut_ptr(), 3) };
        let mut value = 0.0;
        let status = unsafe { eindir_objective_eval(obj, x_t, &mut value) };
        assert_eq!(status, eindir_status_t::EINDIR_SUCCESS);
        assert_eq!(value, 6.0);
        unsafe {
            tensor_free(x_t);
            eindir_objective_free(obj);
        }
    }

    #[test]
    fn test_objective_grad() {
        let obj = make_test_objective(3);
        let mut x_data = [1.0, 2.0, 3.0];
        let x_t = unsafe { create_borrowed_f64_1d(x_data.as_mut_ptr(), 3) };
        let mut g_data = [0.0f64; 3];
        let g_t = unsafe { create_borrowed_f64_1d(g_data.as_mut_ptr(), 3) };
        let status = unsafe { eindir_objective_grad(obj, x_t, g_t) };
        assert_eq!(status, eindir_status_t::EINDIR_SUCCESS);
        assert_eq!(g_data, [1.0, 1.0, 1.0]);
        unsafe {
            tensor_free(x_t);
            tensor_free(g_t);
            eindir_objective_free(obj);
        }
    }

    #[test]
    fn test_objective_bounds() {
        let obj = make_test_objective(2);
        let mut buf = [0.0f64; 2];
        let t = unsafe { create_borrowed_f64_1d(buf.as_mut_ptr(), 2) };
        let s = unsafe { eindir_objective_bounds_low(obj, t) };
        assert_eq!(s, eindir_status_t::EINDIR_SUCCESS);
        assert_eq!(buf, [-5.0, -5.0]);

        let s = unsafe { eindir_objective_bounds_high(obj, t) };
        assert_eq!(s, eindir_status_t::EINDIR_SUCCESS);
        assert_eq!(buf, [5.0, 5.0]);
        unsafe {
            tensor_free(t);
            eindir_objective_free(obj);
        }
    }

    #[test]
    fn test_null_arguments() {
        assert_eq!(unsafe { eindir_objective_dim(std::ptr::null()) }, 0);
        assert_eq!(unsafe { eindir_objective_has_grad(std::ptr::null()) }, 0);
        unsafe { eindir_objective_free(std::ptr::null_mut()) }; // no-op

        let mut v = 0.0;
        let s = unsafe {
            eindir_objective_eval(std::ptr::null(), std::ptr::null(), &mut v)
        };
        assert_eq!(s, eindir_status_t::EINDIR_INVALID_PARAMETER);
    }

    #[test]
    fn test_no_grad_fn() {
        let dim = 2;
        let mut low = vec![-1.0f64; dim];
        let mut high = vec![1.0f64; dim];
        let low_t = unsafe { create_borrowed_f64_1d(low.as_mut_ptr(), dim) };
        let high_t = unsafe { create_borrowed_f64_1d(high.as_mut_ptr(), dim) };

        let obj = unsafe {
            eindir_objective_new(
                dim,
                low_t,
                high_t,
                sum_eval,
                None, // no grad
                std::ptr::null_mut(),
                None,
            )
        };
        unsafe {
            tensor_free(low_t);
            tensor_free(high_t);
        }

        assert_eq!(unsafe { eindir_objective_has_grad(obj) }, 0);

        // Trying to compute grad should return INVALID_PARAMETER
        let mut x_data = [0.5, 0.5];
        let x_t = unsafe { create_borrowed_f64_1d(x_data.as_mut_ptr(), 2) };
        let mut g_data = [0.0f64; 2];
        let g_t = unsafe { create_borrowed_f64_1d(g_data.as_mut_ptr(), 2) };
        let s = unsafe { eindir_objective_grad(obj, x_t, g_t) };
        assert_eq!(s, eindir_status_t::EINDIR_INVALID_PARAMETER);
        unsafe {
            tensor_free(x_t);
            tensor_free(g_t);
            eindir_objective_free(obj);
        }
    }

    #[test]
    fn test_rust_trait_eval() {
        let obj = make_test_objective(3);
        let o = unsafe { &*obj };
        // Use Objective::eval from Rust
        let x = ndarray::array![1.0, 2.0, 3.0];
        let val = o.eval(x.view());
        assert_eq!(val, 6.0);
        unsafe { eindir_objective_free(obj) };
    }

    #[test]
    fn test_rust_trait_grad() {
        let obj = make_test_objective(3);
        let o = unsafe { &*obj };
        let x = ndarray::array![1.0, 2.0, 3.0];
        let g = o.grad(x.view());
        assert_eq!(g, ndarray::array![1.0, 1.0, 1.0]);
        unsafe { eindir_objective_free(obj) };
    }
}
