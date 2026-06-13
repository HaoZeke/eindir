//! C ABI surface for eindir-core.
//!
//! `eindir_objective_t` is `#[repr(C)]` so consumers (e.g. rgpot-core) can embed
//! it as the first member of a derived struct and cast between pointer types at
//! zero cost -- the C "is-a" pattern. Keep non-C-ABI fields such as `Vec`,
//! `Box`, and `OnceLock` out of this struct; use raw pointers and manage
//! lifetimes in the corresponding constructors and destructors.
//!
//! Heap-allocated lifecycle:
//!
//! - create an objective with `eindir_objective_new`;
//! - evaluate it with `eindir_objective_eval`;
//! - release it with `eindir_objective_free`.
//!
//! Embedded lifecycle:
//!
//! - place `eindir_objective_t` as the first member of the consumer struct;
//! - cast the consumer pointer to `eindir_objective_t*` for evaluation;
//! - call the consumer's own destructor rather than `eindir_objective_free`.

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::{c_char, c_void};

use dlpk::sys::DLManagedTensorVersioned;

// ---------------------------------------------------------------------------
// Version
// ---------------------------------------------------------------------------

/// Returns the eindir-core package version as a NUL-terminated ASCII string.
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
    EINDIR_SUCCESS = 0,
    EINDIR_INVALID_PARAMETER = 1,
    EINDIR_INTERNAL_ERROR = 2,
}

// ---------------------------------------------------------------------------
// Thread-local error message
// ---------------------------------------------------------------------------

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

pub fn set_last_error(msg: &str) {
    LAST_ERROR.with(|cell| {
        let c = CString::new(msg)
            .unwrap_or_else(|_| CString::new("(error message contained interior NUL)").unwrap());
        *cell.borrow_mut() = c;
    });
}

/// Retrieve the last error message for the current thread.
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
pub type EindirEvalFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> eindir_status_t;

/// Callback for evaluating the gradient (NULL = no analytic gradient).
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
// Embeddable objective handle
// ---------------------------------------------------------------------------

/// Embeddable, C-ABI-compatible objective handle.
///
/// `#[repr(C)]` means a consumer can embed this struct as the first member
/// of a derived struct and cast `DerivedStruct*` to `eindir_objective_t*` at
/// zero cost. Keep Rust-specific types such as `Vec`, `Box`, and `OnceLock` out
/// of this layout.
///
/// Lifecycle:
///
/// - Created by [`eindir_objective_new`]: `low` and `high` are heap-allocated
///   (len = `dim`) and freed by [`eindir_objective_free`].
/// - Embedded: the consumer fills the fields directly and calls its own free
///   function; never call [`eindir_objective_free`] on an embedded base.
#[repr(C)]
pub struct eindir_objective_t {
    /// Number of input dimensions.
    pub dim: usize,
    /// Lower bounds: heap-allocated f64[dim], owned by this struct when
    /// created via eindir_objective_new.
    pub low: *mut f64,
    /// Upper bounds: heap-allocated f64[dim], owned by this struct when
    /// created via eindir_objective_new.
    pub high: *mut f64,
    /// Value callback.  Must not be NULL.
    pub eval_fn: EindirEvalFn,
    /// Gradient callback.  NULL when no analytic gradient is available.
    pub grad_fn: EindirGradFn,
    /// Opaque context forwarded verbatim to every callback invocation.
    pub user_data: *mut c_void,
    /// Optional destructor for `user_data`; called by eindir_objective_free.
    /// Set to NULL when embedding; manage cleanup in the derived struct's own
    /// free function.
    pub free_fn: EindirFreeFn,
}

// Safety: user_data pointer is opaque; caller guarantees thread safety.
unsafe impl Send for eindir_objective_t {}
unsafe impl Sync for eindir_objective_t {}

// ---------------------------------------------------------------------------
// DLPack tensor helpers (internal)
// ---------------------------------------------------------------------------

/// Create a non-owning 1-D f64 DLPack tensor wrapping `data`.
///
/// # Safety
/// `data` must point to at least `len` f64 values that outlive the tensor.
pub(crate) unsafe fn create_borrowed_f64_1d(
    data: *mut f64,
    len: usize,
) -> *mut DLManagedTensorVersioned {
    use dlpk::sys::{DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLPackVersion, DLTensor};

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

/// Free a DLPack tensor by calling its deleter.
pub(crate) unsafe fn tensor_free(tensor: *mut DLManagedTensorVersioned) {
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

/// Create a new heap-allocated objective handle from C callbacks.
///
/// Bounds data is copied from the DLPack tensors; the caller may free them
/// after this call returns.  Returns NULL on failure.
///
/// The caller must eventually pass the returned pointer to
/// [`eindir_objective_free`].
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

    let low_vec = match copy_bounds(bounds_low, "bounds_low") {
        Some(v) => v,
        None => return std::ptr::null_mut(),
    };
    let high_vec = match copy_bounds(bounds_high, "bounds_high") {
        Some(v) => v,
        None => return std::ptr::null_mut(),
    };

    // Leak the Vecs into raw pointers -- eindir_objective_free reclaims them.
    let low = {
        let mut v = low_vec;
        let ptr = v.as_mut_ptr();
        std::mem::forget(v);
        ptr
    };
    let high = {
        let mut v = high_vec;
        let ptr = v.as_mut_ptr();
        std::mem::forget(v);
        ptr
    };

    Box::into_raw(Box::new(eindir_objective_t {
        dim,
        low,
        high,
        eval_fn,
        grad_fn,
        user_data,
        free_fn,
    }))
}

/// Free an objective handle created by [`eindir_objective_new`].
///
/// Calls `free_fn(user_data)` if a destructor was registered, frees the bounds
/// arrays, then frees the struct.  Do NOT call this on embedded objectives;
/// call the owning struct's free function (e.g. `rgpot_potential_free`) instead.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eindir_objective_free(obj: *mut eindir_objective_t) {
    if obj.is_null() {
        return;
    }
    let o = unsafe { &*obj };
    if let Some(ff) = o.free_fn {
        if !o.user_data.is_null() {
            unsafe { ff(o.user_data) };
        }
    }
    // Reclaim the bounds arrays allocated in eindir_objective_new.
    if !o.low.is_null() {
        unsafe { drop(Vec::from_raw_parts(o.low, o.dim, o.dim)) };
    }
    if !o.high.is_null() {
        unsafe { drop(Vec::from_raw_parts(o.high, o.dim, o.dim)) };
    }
    unsafe { drop(Box::from_raw(obj)) };
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

/// Returns the number of input dimensions.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eindir_objective_dim(obj: *const eindir_objective_t) -> usize {
    if obj.is_null() {
        return 0;
    }
    unsafe { (*obj).dim }
}

/// Writes the lower-bound vector into `out` (pre-allocated DLPack tensor, shape [dim]).
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
        unsafe { std::ptr::copy_nonoverlapping(o.low, dst, o.dim) };
        eindir_status_t::EINDIR_SUCCESS
    }))
}

/// Writes the upper-bound vector into `out` (pre-allocated DLPack tensor, shape [dim]).
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
        unsafe { std::ptr::copy_nonoverlapping(o.high, dst, o.dim) };
        eindir_status_t::EINDIR_SUCCESS
    }))
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// Evaluate the objective at point `x`.
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
/// Returns `EINDIR_INVALID_PARAMETER` when no gradient callback was registered.
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

/// Returns non-zero if the objective has a gradient callback.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eindir_objective_has_grad(obj: *const eindir_objective_t) -> i32 {
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
// Rust-side wrapper for trait use
//
// eindir_objective_t is #[repr(C)] and must not store Rust-specific types
// (Bounds, OnceLock, etc.).  EindirObjectiveWrapper borrows an
// eindir_objective_t and provides the Objective<f64> / Gradient<f64> surfaces.
// ---------------------------------------------------------------------------

use crate::{Bounds, Objective};
use ndarray::{Array1, ArrayView1};

/// Rust-side view over a C objective, implementing Objective/Gradient.
pub struct EindirObjectiveWrapper<'a> {
    obj: &'a eindir_objective_t,
    bounds: Bounds<f64>,
}

impl<'a> EindirObjectiveWrapper<'a> {
    /// # Safety
    /// `obj.low` and `obj.high` must point to `obj.dim` valid f64 values.
    pub unsafe fn new(obj: &'a eindir_objective_t) -> Self {
        let low = unsafe { std::slice::from_raw_parts(obj.low, obj.dim) };
        let high = unsafe { std::slice::from_raw_parts(obj.high, obj.dim) };
        EindirObjectiveWrapper {
            obj,
            bounds: Bounds::new(
                Array1::from_vec(low.to_vec()),
                Array1::from_vec(high.to_vec()),
                0.0,
            ),
        }
    }
}

impl Objective<f64> for EindirObjectiveWrapper<'_> {
    fn dim(&self) -> usize {
        self.obj.dim
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let mut data = x.to_owned();
        let len = data.len();
        let tensor = unsafe { create_borrowed_f64_1d(data.as_mut_ptr(), len) };
        let mut value: f64 = f64::NAN;
        let status = unsafe { (self.obj.eval_fn)(self.obj.user_data, tensor, &mut value) };
        unsafe { tensor_free(tensor) };
        if status != eindir_status_t::EINDIR_SUCCESS {
            return f64::NAN;
        }
        value
    }
}

use crate::gradient::Gradient;

impl Gradient<f64> for EindirObjectiveWrapper<'_> {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let grad_fn = match self.obj.grad_fn {
            Some(gf) => gf,
            None => return Array1::zeros(self.obj.dim),
        };
        let mut x_data = x.to_owned();
        let len = x_data.len();
        let x_tensor = unsafe { create_borrowed_f64_1d(x_data.as_mut_ptr(), len) };

        let mut grad_data = vec![0.0f64; self.obj.dim];
        let grad_tensor = unsafe { create_borrowed_f64_1d(grad_data.as_mut_ptr(), self.obj.dim) };

        let _status = unsafe { grad_fn(self.obj.user_data, x_tensor, grad_tensor) };
        unsafe {
            tensor_free(x_tensor);
            tensor_free(grad_tensor);
        }
        Array1::from_vec(grad_data)
    }

    fn dim(&self) -> usize {
        self.obj.dim
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
        unsafe { eindir_objective_free(std::ptr::null_mut()) };

        let mut v = 0.0;
        let s = unsafe { eindir_objective_eval(std::ptr::null(), std::ptr::null(), &mut v) };
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
                None,
                std::ptr::null_mut(),
                None,
            )
        };
        unsafe {
            tensor_free(low_t);
            tensor_free(high_t);
        }

        assert_eq!(unsafe { eindir_objective_has_grad(obj) }, 0);

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
        let wrapper = unsafe { EindirObjectiveWrapper::new(&*obj) };
        let x = ndarray::array![1.0, 2.0, 3.0];
        let val = wrapper.eval(x.view());
        assert_eq!(val, 6.0);
        unsafe { eindir_objective_free(obj) };
    }

    #[test]
    fn test_rust_trait_grad() {
        let obj = make_test_objective(3);
        let wrapper = unsafe { EindirObjectiveWrapper::new(&*obj) };
        let x = ndarray::array![1.0, 2.0, 3.0];
        let g = wrapper.grad(x.view());
        assert_eq!(g, ndarray::array![1.0, 1.0, 1.0]);
        unsafe { eindir_objective_free(obj) };
    }

    /// Verify ABI layout: first field at offset 0 (required for embedding).
    #[test]
    fn eindir_objective_t_is_repr_c() {
        use std::mem::offset_of;
        assert_eq!(offset_of!(eindir_objective_t, dim), 0);
        assert_eq!(
            offset_of!(eindir_objective_t, low),
            std::mem::size_of::<usize>()
        );
    }
}
