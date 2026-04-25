//! C ABI surface. Filled in once the typed algebra lands; the only export
//! today is the package version string for downstream sanity checks.

use std::os::raw::c_char;

#[unsafe(no_mangle)]
pub extern "C" fn eindir_core_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}
