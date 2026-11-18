#pragma once

/**
 * \file eindir-core.hpp
 * \brief C++ companion header for eindir-core.
 *
 * Hand-written companion to the cbindgen-generated `eindir-core.h`. The C
 * header is usable from C++ on its own; this file adds C++ helpers in the
 * `eindir` namespace. Both need the DLPack 1.0 headers on the include path,
 * because `eindir-core.h` includes `<dlpack/dlpack.h>`.
 *
 * The objective handles, the ABI stamp and the descriptor checks have no
 * C++ wrappers; call the C functions directly.
 */

#include "eindir-core.h"

/// C++ helpers over the eindir C ABI.
namespace eindir {

/// Returns the eindir-core package version as a NUL-terminated C string.
///
/// The same string as `eindir_core_version()`, with static lifetime.
inline const char* version() noexcept {
    return eindir_core_version();
}

}  // namespace eindir
