# AGENTS.md

Working notes for changing or scripting against eindir.
The manual lives in `docs/orgmode/`.
`llms.txt` indexes it, and `llms-full.txt` has all of it in one file, with this file and the skills.
Task guides with scripts that CI runs are under `skills/`.

## What eindir provides

`eindir-core` is a Rust crate (library name `eindir_core`) of primitives for optimisation and sampling within box bounds on R^D.
The anneal optimiser is built on it.

- Objectives within box bounds: `Objective<T>` over `Bounds<T>`, with `eval`, `eval_batch`, and `eval_batch_parallel`, which uses rayon from `EVAL_BATCH_PARALLEL_MIN` points up.
- Halton point sets: `low_discrepancy_points`, `shifted_low_discrepancy_points`, `boundary_anchored_low_discrepancy_points`, `halton_points`, `halton_unit`.
- Gradients: the `Gradient<T>` trait, `AnalyticGradient`, `FiniteDiffGradient`, `DifferentiableObjective` and, with the `autodiff` feature, `ForwardAd`.
- A thermostat for the generalized Langevin equation (GLE): `GleThermostat`, `optimal_sampling_drift`, `matrix_exp`, `ldl_sqrt`.
- Surrogates: `AdditiveSurrogate`, `ChebyshevSurrogate`, `ReducedObjective`.
- Test functions: `Rastrigin`, `Rosenbrock`, `Ackley`, `StybTang2D`.

Bindings:

- Python: the `eindir` package, with the extension module `eindir._core`.
  It covers much less than the crate, and the pitfalls below say what it lacks.
- C: `libeindir_core` and `eindir-core.h`, built with the `capi` feature.
  C++ uses the same header, plus `include/eindir-core.hpp`.

## Layout

- `Cargo.toml`: the crate, its feature flags and the only hand-written version.
- `src/`: the Rust sources, with the C ABI in `ffi.rs` and the `_core` extension in `python.rs` and `py_objective.rs`.
- `python/eindir/`: the pure-Python package (`NumLimit`, `ObjectiveFunction`, array API helpers, `viz`).
- `pytest/`: the pytest suite.
- `tests/`: Rust integration tests, `ffi_smoke.c`, `header_check.c` and `header_check.cpp`.
- `include/`: `eindir-core.hpp`.
- `examples/`: `batch_scale.rs`, which times `eval_batch_parallel` against the thread count.
- `cbindgen.toml`, `meson.build`, `CMakeLists.txt`: header generation and glue for C and C++ consumers.
- `docs/orgmode/`: the manual sources in Org, with the Sphinx settings in `docs/source/conf.py`.
- `docs/newsfragments/`: towncrier fragments for the next release.
- `tools/`: version sync and drift checks.
- `skills/`: task guides, each a `SKILL.md` with `scripts/` and `assets/`.
- `scripts/`: `llms/` writes `llms.txt`, and `skills/` checks and runs the guides.

## Toolchain

- Rust 1.88 or newer (edition 2024).
  An older cargo stops at the manifest with "feature `edition2024` is required", and so do maturin, cbindgen and the version check, which goes through `cargo metadata`.
  Install a current stable, or prefix single commands with `RUSTUP_TOOLCHAIN=stable`.
- Python 3.10 or newer.
  The wheel is abi3 (`cp310`), and building it needs maturin 1.5 or newer, below 2.
- For C: cargo-c 0.10.17 or newer, DLPack 1.0 headers and a C99 compiler.
- [pixi](https://pixi.sh) supplies Rust, Python, maturin and the docs tools.
  Its environments are `default`, `python` and `docs`.

## Build

```bash
cargo build --release                    # Rust library; no C symbols
cargo build --release --features capi    # the same with the C ABI
maturin develop --features python        # eindir and eindir._core into the active virtualenv
cargo cinstall --release --prefix="$PREFIX" --libdir=lib   # C header, libraries, eindir-core.pc
```

## Test

`.github/workflows/ci.yml` runs these commands.

```bash
cargo fmt --all -- --check
cargo clippy --all-features --all-targets -- -D warnings
cargo test
cargo test --all-features
python -m pip install "maturin>=1.5,<2" pytest numpy
maturin develop --features python && pytest pytest/
tools/check-version-drift.sh             # needs cogapp: python3 -m pip install cogapp==3.6.0
```

At commit 8613085, `cargo test` passes 41 tests and `cargo test --all-features` 54.
Pytest passes 18 and skips one unless `array-api-strict` is installed.
The `ffi-smoke` job installs the C library with cargo-c and runs `tests/ffi_smoke.c` under valgrind.
The C ABI skill script does the same for its examples.

`.github/workflows/skills.yml` runs these.

```bash
python scripts/llms/build_llms.py --check                     # llms.txt and llms-full.txt are current
python -m unittest discover -s scripts/llms -p "test_*.py"     # the generator
python scripts/skills/run_skills.py                           # every SKILL.md, then its examples
```

`run_skills.py` runs each skill's code with the Python that runs it.
That Python needs eindir (`maturin develop --features python`) and cogapp.
Without cogapp, the contributing skill skips its version check.
The Rust and C skills also need cargo, rustfmt, clippy, cargo-c, and C and C++ compilers.
When valgrind is on PATH, the C ABI skill runs its examples under it as well.

## Docs

```bash
pixi run -e docs docbld    # Org to RST (orgbld), then Sphinx HTML in docs/build/
```

`docs/source/*.rst` is generated and ignored by git, so edit the Org files.
Emacs installs ox-rst from Milkypostman's Emacs Lisp Package Archive (MELPA) the first time, so the first build needs network access.
After changing the manual, this file or a skill, run `python3 scripts/llms/build_llms.py` and commit the two files it rewrites.

## Common tasks

- Python against the package: `skills/eindir-python/`.
- Rust against the crate: `skills/eindir-rust/`.
- C or C++ against the library, or a C objective for Rust drivers: `skills/eindir-c-abi/`.
- Preparing a commit, fragments, the docs build and version bumps: `skills/eindir-contributing/`.
  Its `scripts/preflight.py --base origin/main` checks the fragments, the llms files, the version copies and the commit subjects.
- A new Python export: the binding in `src/python.rs`, the re-export in `python/eindir/__init__.py` if it belongs in `eindir`, a test in `pytest/`, the expected surface in `skills/eindir-python/scripts/surface.py`, the Org page, and a fragment.
- A version bump: edit `version` in `Cargo.toml`, then run `tools/sync-version.sh`, `cargo update --workspace` and `tools/check-version-drift.sh`.
- A fragment: `docs/newsfragments/+<slug>.<type>.rst`, where `<type>` is one of `added`, `changed`, `fixed`, `removed`, `deprecated`, `security`, `dev` or `misc`.
  Code in a fragment takes double backticks, as in reStructuredText.
  Towncrier skips a file with an unknown type or in a subdirectory, without a warning.
  It also links an issue to a file whose name lacks the `+`.
  `preflight.py` reports all three.

## Pitfalls

Python, each reproduced by `skills/eindir-python/scripts/pitfalls.py`:

- The Python examples on the Quickstart page and in four tutorials (quickstart, low-discrepancy, GLE thermostat, surrogates) stop with ImportError, against main and against PyPI 0.5.1 alike.
  They import names that do not exist: `Bounds` and `Objective` from `eindir`, `GleThermostat` and `AdditiveSurrogate`.
  `skills/eindir-python/` shows calls that work.
- `GleThermostat`, `optimal_sampling_drift`, `matrix_exp`, `ldl_sqrt`, `halton_unit`, the surrogates, the gradient adapters and the test-function types such as `Rastrigin` are Rust-only.
  `eindir._core.Bounds` and `PyObjective` exist, but `eindir.Bounds` does not.
- Read the version from `eindir._core.__version__`, since `eindir.__version__` does not exist.
- `_core` takes contiguous float64 NumPy arrays only: lists, integer arrays and strided views raise TypeError.
- Rust panics reach Python as `pyo3_runtime.PanicException`, which derives from BaseException, so `except Exception` misses it.
  Three calls panic: `_core.Bounds` with `low > high`, `Bounds.clip` with a point of the wrong length, and a `PyObjective` called from another thread.
- eindir 0.5.1 from PyPI does not check bounds: `_core.Bounds` accepts `low > high`, and `contains` and `clip` look only at the coordinates the point has.
  The checks came in 0.6.0.
- `PyObjective.eval` does not check the length of the point, and `PyObjective.grad` without `grad_fn` raises ValueError.
  Python has no finite-difference fallback.
- `NumLimit(low, high)` keeps `dims=1` unless told otherwise.
- `pip install "eindir[all]"` installs no extras, since the only extras are `plotter` and `testing`.

Rust (see `skills/eindir-rust/`):

- `FiniteDiffGradient` uses central differences with `h = 1e-5`, 2·dim evaluations per gradient.
- On a type that implements both `Objective` and `Gradient`, `x.dim()` is ambiguous (E0034), so write `Objective::dim(&x)`.
- `DifferentiableObjective` has no blanket impl, so add an empty `impl DifferentiableObjective<f64> for T {}`.
- `Bounds::new` panics on lengths that differ, on a negative slack and on `low > high`.
  `mkpoint` panics when a coordinate has `low == high`.
- `GleThermostat::canonical(drift, dt, temperature, kb)` takes Boltzmann's constant as its fourth argument and has no mass argument.

C (see `skills/eindir-c-abi/`):

- Without the `capi` feature the library exports no `eindir_` symbols.
  cargo-c enables it by itself.
- A program linked against `target/<triple>/release` from `cargo cbuild` fails to start: the library's soname is `libeindir_core.so.0.7` and no file of that name is there.
  Install with `cargo cinstall` and link the prefix.
- Ubuntu's `libdlpack-dev` is DLPack 0.6, which lacks `DLManagedTensorVersioned`.
  Use the v1.0 headers pinned in `ci.yml`.
- eindir passes tensors to the callbacks unchecked.
  A failing callback leaves `eindir_last_error()` as it was, and a success does not clear it.
- `schema_id` in a required descriptor is never a wildcard, although the header says empty fields are.

Repository:

- `cog` on PATH may be cocogitto, the commit checker.
  The version scripts need cogapp and call it as `python3 -m cogapp`.
- cocogitto `6.3.0` reads `cog.toml`, and `7.0.0` refuses it ("unknown field `packages`").
- At commit 8613085 the Lint job fails at `cargo fmt --check` (11 hunks in 6 files).
  After it, clippy would fail with 4 errors, and `cog check` on two `REL:` commits from 2024.
  The other 16 checks pass.
- With `-Dwith_examples=true`, `meson.build` calls `subdir('examples')`, but `examples/` has no `meson.build`.

Releases and the site, as of 2026-11-03:

- `Cargo.toml` on main says 0.7.0, crates.io has 0.6.0, PyPI has 0.5.1 and the newest git tag is v0.5.2.
  Check `cargo info eindir-core` and `pip index versions eindir` before relying on a released feature.
- The HTML pages of [the site](https://eindir.rgoswami.me/) answer scripted clients with HTTP 403 and a challenge page.
  A path that does not exist returns the front page with status 200.
  Page sources are readable at `/_sources/<page>.rst.txt`, and `llms-full.txt` in the repository has all of them.
