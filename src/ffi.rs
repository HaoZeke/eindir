//! C ABI surface. Filled in once the typed algebra lands; the only export
//! today is the package version string for downstream sanity checks.

use std::os::raw::c_char;

/// Returns the eindir-core package version as a NUL-terminated ASCII string.
///
/// The returned pointer is valid for the lifetime of the process - the string
/// lives in the binary's read-only data segment and is never freed.
#[unsafe(no_mangle)]
pub extern "C" fn eindir_core_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}
