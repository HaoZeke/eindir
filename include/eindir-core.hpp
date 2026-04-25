#pragma once

/**
 * \file eindir-core.hpp
 * \brief C++ companion header for eindir-core.
 *
 * Hand-written companion to the cbindgen-generated `eindir-core.h`. Wraps
 * the C ABI exports in the `eindir` namespace for ergonomic C++ usage.
 */

#include "eindir-core.h"

namespace eindir {

/// Returns the eindir-core package version as a NUL-terminated C string.
inline const char* version() noexcept {
    return ::eindir_core_version();
}

}  // namespace eindir
