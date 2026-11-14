![Logo](./branding/logo/eindir_logo.png)

# eindir [![DOI](https://zenodo.org/badge/605541458.svg)](https://zenodo.org/doi/10.5281/zenodo.10672737)


A set of "particles" or components mainly focused on working with functions in ND.

The Rust crate `eindir-core` has the primitives: objectives over box bounds, Halton point sets, gradients, a coloured-noise thermostat for the generalized Langevin equation (GLE) and Chebyshev surrogates.
The Python package `eindir` covers the bounds, the objectives and the point sets, and a C library with a C++ header lets native codes pass objectives through the same interface.

## Quick start

Python, after `pip install eindir`, or `maturin develop --features python` in a checkout:

```python
import numpy as np
import eindir
from eindir import _core

low = np.array([-5.12, -5.12])
high = np.array([5.12, 5.12])
obj = _core.PyObjective(lambda x: float(np.sum((x - 1.0) ** 2)), _core.Bounds(low, high))

starts = eindir.low_discrepancy_points(low, high, 8)
values = [obj.eval(row) for row in starts]
best = int(np.argmin(values))
print(starts[best], values[best])
```

```text
[1.28       2.84444444] 3.480375308641971
```

Rust, with `eindir-core = "0.6"` and `ndarray = "0.17"` under `[dependencies]`, or `eindir-core = { git = "https://github.com/HaoZeke/eindir" }` for the main branch:

```rust
use eindir_core::{Bounds, Objective, low_discrepancy_points};
use ndarray::{ArrayView1, array};

struct Sphere {
    bounds: Bounds<f64>,
}

impl Objective<f64> for Sphere {
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        x.iter().map(|xi| (xi - 1.0).powi(2)).sum()
    }
}

fn main() {
    let bounds = Bounds::new(array![-5.12, -5.12], array![5.12, 5.12], 1e-6);
    let obj = Sphere { bounds };
    let starts = low_discrepancy_points(obj.bounds(), 8, 1);
    let values = obj.eval_batch(starts.view());
    let best = (0..values.len())
        .min_by(|&a, &b| values[a].total_cmp(&values[b]))
        .unwrap();
    println!("{} {}", starts.row(best), values[best]);
}
```

```text
[1.2800000000000002, 2.844444444444443] 3.480375308641971
```

Both draw the same Halton points.
The GLE thermostat, the surrogates and the gradient adapters have no Python binding.
The Python examples on the Quickstart page and in four tutorials stop with ImportError.
The [Python skill](https://github.com/HaoZeke/eindir/blob/main/skills/eindir-python/SKILL.md) lists what `eindir` exports.

## Documentation

The manual is at https://eindir.rgoswami.me/.
Its sources are the Org files in `docs/orgmode/`, and `pixi run -e docs docbld` builds it into `docs/build/`.

- [llms.txt](https://raw.githubusercontent.com/HaoZeke/eindir/main/llms.txt) indexes the manual, and [llms-full.txt](https://raw.githubusercontent.com/HaoZeke/eindir/main/llms-full.txt) has every page in one Markdown file, with `AGENTS.md` and the skills.
  `scripts/llms/build_llms.py` writes both.
- [AGENTS.md](https://github.com/HaoZeke/eindir/blob/main/AGENTS.md) has the build, test and docs commands as CI runs them, and the mistakes that are easy to make with each binding.
- [skills/](https://github.com/HaoZeke/eindir/tree/main/skills) has task guides for Python, Rust, the C ABI and making a change.
  CI runs their examples.

## Development

The crate needs Rust 1.88 or newer (edition 2024).
The tests as CI runs them, with the Python steps in a virtualenv:

``` sh
cargo test --all-features
python -m pip install "maturin>=1.5,<2" pytest numpy
maturin develop --features python
pytest pytest/
```

With pixi, `pixi run test` runs the Rust tests and `pixi run -e python python-test` builds the extension and runs pytest.
Each change adds a `towncrier` fragment to `docs/newsfragments/`.
The version is set in `Cargo.toml` alone, and `tools/sync-version.sh` copies it to the other files.

## License
MIT

## Relation to anneal

eindir supplies the Objective trait, Bounds, low-discrepancy generators (Halton, etc.), GLE thermostat matrices, surrogate primitives (AdditiveSurrogate, Chebyshev, ReducedObjective), and an optional `autodiff` feature (`num-dual` forward mode).
These primitives let anneal and downstream codes stay inside the typed five-component algebra and share one implementation of each slot.

**User entry for optimization:** start with [anneal](https://github.com/HaoZeke/anneal) (`pip install anneal`), the budget-only `global_optimize` API, the notebook `examples/notebooks/01_quickstart.ipynb`, and https://anneal.rgoswami.me — not the eindir trait surface.

Project history: continuous development since **2023-02**.
Paper reproducibility: [anneal_repro](https://github.com/HaoZeke/anneal_repro), Zenodo [10.5281/zenodo.20672621](https://doi.org/10.5281/zenodo.20672621).

## Native compatibility contract

The C ABI exposes two complementary contracts for consumers such as rgpot and xtsci-optimize:

1. `eindir_core_abi_stamp()` reports the native handle layout.
   Consumers must require the same `abi_major`, `objective_layout`, `objective_size`, and `objective_align`.
   `abi_minor` and the DLPack minor version may be less than or equal to the consumer's supported value.
   Required feature bits must be a subset of the advertised `features`.
2. `eindir_objective_descriptor()` reports the semantic objective contract.
   `eindir_objective_descriptor_compatible(actual, required)` checks schema, producer, units, energy and gradient signs, operations, DLPack device and dtype, tensor layout, and callback lifetime.
   Zero scalar requirements and empty or NULL strings are wildcards, except `schema_id`, which must be set and equal on both sides.
   Operation bits are checked as a required subset.

`eindir_core_abi_stamp()` returns ABI major `1`, ABI minor `1`, objective layout `3`, DLPack `1.0`, and the gradient and batch feature bits.
The stable family string is `eindir.objective`.
Consumers should call the validation functions before the first objective evaluation and include the received `eindir_abi_stamp_t` or descriptor in a rejection diagnostic.
