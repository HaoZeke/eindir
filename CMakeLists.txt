# CMake glue for eindir-core.
#
# Mirrors the meson script: drives the Rust crate build through Corrosion
# (so cargo handles every transitive crate dep), regenerates the public
# C header via cbindgen, and exposes an `eindir-core::eindir-core` target
# that downstream `find_package`/`add_subdirectory` consumers can link
# against. The cargo-c-installed pkg-config file (`eindir-core.pc`) gives
# system-wide consumers an alternate, install-time entry point.
cmake_minimum_required(VERSION 3.22)
project(eindir-core VERSION 0.2.0 LANGUAGES C CXX)

# Corrosion builds the Rust crate from Cargo.toml
include(FetchContent)
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5
)
FetchContent_MakeAvailable(Corrosion)

corrosion_import_crate(MANIFEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
                       CRATES eindir-core)
# Corrosion >=0.5 mirrors cargo's post-1.79 dash-to-underscore rule for
# library target names, so the imported target is `eindir_core` even though
# the package name is `eindir-core`.
corrosion_set_env_vars(eindir_core "CARGO_CFG_TARGET_OS=$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>")

# Generate the C header via cbindgen
find_program(CBINDGEN cbindgen REQUIRED)
set(EINDIR_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/eindir-core.h)
add_custom_command(
    OUTPUT ${EINDIR_HEADER}
    COMMAND ${CBINDGEN}
        -q
        --config ${CMAKE_CURRENT_SOURCE_DIR}/cbindgen.toml
        --crate eindir
        --output ${EINDIR_HEADER}
        -- ${CMAKE_CURRENT_SOURCE_DIR}/src/lib.rs
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cbindgen.toml
            ${CMAKE_CURRENT_SOURCE_DIR}/src/lib.rs
    COMMENT "Generating eindir-core.h via cbindgen"
)
add_custom_target(eindir-core-header DEPENDS ${EINDIR_HEADER})
add_dependencies(eindir_core eindir-core-header)

# Exported target for consumers: eindir-core::eindir-core
add_library(eindir-core::eindir-core ALIAS eindir_core)

target_include_directories(eindir_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Install header alongside the library
install(FILES ${EINDIR_HEADER} DESTINATION include)
