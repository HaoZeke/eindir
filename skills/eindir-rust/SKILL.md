---
name: eindir-rust
description: Write Rust against the eindir-core crate. Implement the Objective trait on box Bounds, draw Halton start points, evaluate batches in parallel, and supply gradients (analytic, central finite differences or forward-mode autodiff). Covers the builtin test objectives, the generalized Langevin equation (GLE) thermostat with coloured noise built from the optimal-sampling drift, and fitting and sampling the additive Chebyshev surrogate. Use for the GLE thermostat and the surrogates, which have no Python binding, when the eindir manual's Python examples for them fail, or when eindir-core does not build or panics.
license: See LICENSE in the eindir repository
compatibility: Rust 1.88 or newer (edition 2024) with cargo. The example script needs Python 3.11 or newer, rustfmt, clippy and crates.io access the first time.
---

# eindir-core from Rust

## Depend on it

The package is `eindir-core` and the library is `eindir_core`.
Take `ndarray` and `rand` at the versions `eindir-core` uses, so that `ArrayView1` and the random number generators are the same types:

```toml
[dependencies]
eindir-core = { git = "https://github.com/HaoZeke/eindir" }
ndarray = "0.17"
rand = "0.9"
```

crates.io releases can lag the main branch, so compare `cargo info eindir-core` with `version` in the repository's `Cargo.toml` before choosing one.

The crate uses edition 2024 and `rust-version = "1.88"`.
An older cargo stops at the manifest with "feature `edition2024` is required".
With rustup, use `rustup run stable cargo ...` or set `RUSTUP_TOOLCHAIN=stable`.

## An objective

`Objective<T>` needs `dim`, `bounds` and `eval`.
`eval_batch` (serial by default) and `global_min` are optional.
The trait requires `Send + Sync`, so keep counters in atomics or a `Mutex`, not `Cell` or `RefCell`.
From [assets/objective_tour.rs](assets/objective_tour.rs):

```rust
impl Objective<f64> for ShiftedSphere {
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.evals.fetch_add(1, Ordering::Relaxed);
        x.iter().map(|xi| (xi - 1.0).powi(2)).sum()
    }
}
```

The trait requires `bounds` to return a `Bounds<T>`.
`Bounds::new(low, high, slack)` panics when the lengths differ, when `slack` is negative or when any `low[i] > high[i]`.
`contains` allows `slack` beyond each face and returns false for a point of the wrong length, and `clip` panics on one.
`mkpoint` (uniform sampling) panics on a coordinate with `low[i] == high[i]`.

## Start points and batches

```rust
let starts = low_discrepancy_points(obj.bounds(), 16, 1);
assert_eq!(starts.dim(), (16, 3));
assert!(starts.outer_iter().all(|row| obj.bounds().contains(row)));

let serial = obj.eval_batch(starts.view());
let parallel = eval_batch_parallel(&obj, starts.view());
assert_eq!(serial, parallel);
```

`low_discrepancy_points(bounds, n, skip)` is the Halton set (`halton_points`) scaled to the box, one point per row.
`skip = 0` puts the low corner first.
`shifted_low_discrepancy_points(bounds, n, skip, seed)` adds a seeded random shift, one independent replica per seed, and `boundary_anchored_low_discrepancy_points` puts the box vertices first when `n >= 2^dim`.
`eval_batch_parallel` spreads the rows over rayon once there are `EVAL_BATCH_PARALLEL_MIN` of them and calls `eval_batch` below that.

## Gradients

`Gradient<T>` has `grad` and `dim`.
`AnalyticGradient::new(dim, f)` wraps a closure.
`FiniteDiffGradient::new(obj)` takes ownership of an objective and uses central differences with `h = 1e-5` (`with_step(obj, h)` to change it).
That costs `2 * dim` evaluations per gradient.
The n_p+1 that the manual's gradients and FAQ pages give for a finite-difference fallback is not this type's cost.

```rust
let fd = FiniteDiffGradient::new(obj);
let before = fd.obj.evals.load(Ordering::Relaxed);
let x = array![0.0, 2.0, 3.0];
let g = fd.grad(x.view());
let used = fd.obj.evals.load(Ordering::Relaxed) - before;
assert_eq!(used, 2 * x.len());
```

`DifferentiableObjective<T>` (an `Objective` and a `Gradient`, with `value_and_gradient`) has no blanket impl: write `impl DifferentiableObjective<f64> for MyType {}` or override `value_and_gradient` for a fused evaluation.
The builtins `Rastrigin::<D>`, `Rosenbrock::<D>`, `Ackley::<D>` and `StybTang2D` implement all three and return their minimum from `global_min`.
On a type with both traits `x.dim()` is ambiguous (error E0034), so write `Objective::dim(&x)`.
The `autodiff` feature adds `ForwardAd`, which turns a `DualField` written once over `num-dual` into an objective with an exact gradient.

## The GLE thermostat

From [assets/gle_and_surrogate.rs](assets/gle_and_surrogate.rs):

```rust
let omega0 = 0.2;
let drift = optimal_sampling_drift(omega0);
let n = OPTIMAL_SAMPLING_NS + 1;
assert_eq!(drift.dim(), (n, n));

let (dt, temperature, kb) = (0.01, 1.0, 1.0);
let gle = GleThermostat::canonical(&drift, dt, temperature, kb);
let covariance = Array2::<f64>::eye(gle.ns + 1) * temperature;
let dim = 4;
let mut state = gle.sample_stationary(&covariance, dim, kb, &mut rng);
assert_eq!(state.dim(), (gle.ns + 1, dim));
```

`optimal_sampling_drift(omega0)` is the fitted drift for the band `[omega0, 100 omega0]`, with `OPTIMAL_SAMPLING_NS` auxiliary momenta.
`canonical(drift, dt, temperature, kb)` sets the covariance to `temperature * I`.
Its fourth argument is the Boltzmann constant, 1 in reduced units.
Use `GleThermostat::new(drift, covariance, dt, kb)` for any other covariance.
The state is `(ns + 1) x dim`: row 0 is the mass-scaled physical momentum of each coordinate and the other rows are the auxiliary momenta.
`step(&mut state.view_mut(), &mut rng)` applies one `s <- T s + S xi` update in place.
The thermostat never moves positions.
An integrator such as anneal's `gle_langevin` driver moves them.
`matrix_exp` and `ldl_sqrt` are public for building other propagators.

## Surrogates

```rust
let pilot = low_discrepancy_points(&bounds, 128, 1);
let y: Array1<f64> = pilot.outer_iter().map(f).collect();
let degree = 8;
let surrogate = AdditiveSurrogate::fit(pilot.view(), y.view(), bounds.clone(), degree);
```

`AdditiveSurrogate::fit(x, y, bounds, degree)` fits `c + sum_j g_j(x_j)` with a degree-`degree` Chebyshev series per coordinate, by 12 backfitting passes (`fit_backfit` takes the count).
It is an `Objective` and a `Gradient`, and `sample(n, temperature, grid_m, rng)` draws `n` independence proposals inside the box from `exp(-(f - f_min) / T)`.
`ChebyshevSurrogate::new(bounds, terms, coeffs)` evaluates a total-degree Chebyshev series whose multi-indices and coefficients you supply.
No routine in eindir fits them.
`ReducedObjective::new(inner, origin, basis, bounds)` evaluates `inner` at `clip(origin + basis r)` for `r` in the reduced box.

## Run the examples

```bash
RUSTUP_TOOLCHAIN=stable python scripts/run_examples.py
```

`scripts/run_examples.py` builds both files in `assets/` as binaries of a throwaway crate that depends on this checkout (or `$EINDIR_REPO`, or a fresh clone), checks them with rustfmt and `clippy -D warnings`, runs them, and fails if a `rust` or `toml` block in this file no longer matches the examples or `Cargo.toml`.
The objective tour prints the best start, a finite-difference gradient from 6 evaluations and the Rastrigin minimum.
The second example prints the mean squared momentum under the thermostat (close to `kB T`) and the surrogate's worst error on fresh points.

## Features

- `capi`: the C ABI and `eindir-core.h` (see the `eindir-c-abi` skill).
- `py-objective`, `python`: the pyo3 extension `eindir._core`.
- `autodiff`: `ForwardAd` and `DualField` through `num-dual`.

`cargo test` runs the tests of the default features, and `cargo test --all-features` runs the rest.
