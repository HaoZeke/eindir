# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- towncrier release notes start -->

## [0.5.2](https://github.com/HaoZeke/eindir/tree/0.5.2) - 2026-07-18

### Added

- Parallel `eval_batch` and Halton design generation via Rayon, with a
  multi-walker `eval_batch` path sized for population algorithms.

### Fixed

- Documented and hardened the C API FFI contracts (null handling, pointer
  lifetime and length requirements) for the Objective/Gradient surface.


## [0.5.1](https://github.com/HaoZeke/eindir/tree/0.5.1) - 2026-07-10

### Added

- `features` module: `box_geometry` / `BoxGeometry` (dimension, mean width, aspect ratio), `isotropic_proposal_scale`, and `compensated_delta` for numerically safer acceptance-path energy differences.


## [0.5.0](https://github.com/HaoZeke/eindir/tree/0.5.0) - 2026-06-26

### Added

- Expose the Objective/Gradient contract through the cargo-c C API, with
  ``eindir_objective_t`` as a ``repr(C)`` first-member embeddable handle for
  downstream C/C++ consumers.

### Fixed

- Cache the Halton prime table so high-dimensional low-discrepancy designs no
  longer stall on repeated prime generation.
- Correct analytic gradients for the built-in objective functions and split the
  Python objective-handle feature so optional bindings stay coherent.

### Miscellaneous

- Refresh Sphinx/orgmode site docs, repair reference navigation, and stabilize
  the documentation build pipeline.
- Resolve ``dlpk`` from crates.io (v0.1.5) instead of a git pin, unblocking
  registry publication of ``eindir-core``.


## [0.4.7](https://github.com/HaoZeke/eindir/tree/0.4.7) - 2026-06-08

### Added

- Pointset improvements: shifted low-discrepancy replicas, anchored design centers, and anchored low-discrepancy designs.

### Fixed

- Additive finite-weight guard, GLE import cleanups, and Array1 test-local fixes.


## [0.4.4](https://github.com/HaoZeke/eindir/tree/0.4.4) - 2026-06-08

### Added

- Fitted optimal-sampling GLE drift with a benchmarked colored-noise comparison against white noise.


## [0.4.3](https://github.com/HaoZeke/eindir/tree/0.4.3) - 2026-06-08

### Added

- Native GLE colored-noise thermostat with optimal sampling.


## [0.4.2](https://github.com/HaoZeke/eindir/tree/0.4.2) - 2026-06-07

### Added

- Separable rank-1 surrogate (Additive) + tempered independence sampler.


## [0.4.1](https://github.com/HaoZeke/eindir/tree/0.4.1) - 2026-06-07

### Added

- Bounded low-discrepancy designs (pointset API).


## [0.4.0](https://github.com/HaoZeke/eindir/tree/0.4.0) - 2026-06-04

### Added

- ReducedObjective (dimension-collapse via affine encode/decode on retained subspace) and ChebyshevSurrogate (total-degree on reduced box with analytic gradient).
Both are first-class Objective implementations.
Also tvm_ffi tensor interop and Array API namespace helpers.


## [0.3.0](https://github.com/HaoZeke/eindir/tree/0.3.0) - 2026-04-26

### Added

- Objective trait with built-in Ackley, Rastrigin, Rosenbrock, Styblinski-Tang; FPair and Bounds types with proptest law witnesses.
- PyObjective adapter wrapping Python callables into the Objective<f64> algebra.


## [0.2.0](https://github.com/HaoZeke/eindir/tree/0.2.0) - 2026-04-25

### Added

- Pixi workspace with default, python, and docs environments (replacing environment.yml and PDM).
- Replaced legacy MyST + furo docs with orgmode export + Sphinx (shibuya theme) pipeline.
- Scaffolded eindir-core Rust crate (lib + cdylib + staticlib) plus C/C++ bindings via cargo-c, pkg-config, meson, CMake.

### Changed

- Adopted cog for conventional commits + cargo-dist style releases; dropped legacy pre-commit and tbump.
- BREAKING: build system migrated from PDM to maturin mixed mode.
Python sources moved under python/eindir/; wheel ships Rust extension eindir._core.


## [0.1.0](https://github.com/HaoZeke/eindir/tree/0.1.0) - 17-02-2024


No significant changes.


## [0.0.5](https://github.com/HaoZeke/eindir/tree/0.0.5) - 17-02-2024


No significant changes.
