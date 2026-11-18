//! Low-discrepancy point sets for bounded numerical objectives.

use std::sync::{Mutex, OnceLock};

use ndarray::{Array1, Array2, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::Bounds;

/// Rows at which Halton designs fan out across cores.
const HALTON_PARALLEL_MIN: usize = 64;

const SMALL_PRIMES: [u64; 32] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131,
];

/// Lazily extended prime table shared across Halton calls.
///
/// High-dimensional Halton designs ask for one prime per axis. Recomputing
/// the axis-th prime by trial division on every call is quadratic in the
/// dimension and stalls at tens of thousands of axes (a 34k-variable problem
/// needs the ~404000-range prime per axis). The table is grown once on demand
/// and then read in O(1), so a `dim`-dimensional design costs a single sieve
/// pass instead of `dim` independent searches.
fn prime_cache() -> &'static Mutex<Vec<u64>> {
    static CACHE: OnceLock<Mutex<Vec<u64>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(SMALL_PRIMES.to_vec()))
}

fn nth_prime(index: usize) -> u64 {
    if index < SMALL_PRIMES.len() {
        return SMALL_PRIMES[index];
    }
    let mut cache = prime_cache().lock().expect("prime cache poisoned");
    if index >= cache.len() {
        // Trial-divide each new candidate only against the primes already
        // known up to its square root, using `p <= candidate / p` to stay
        // overflow-safe. The table grows to `index + 1` primes exactly once.
        let mut candidate = *cache.last().expect("non-empty prime table") + 2;
        while cache.len() <= index {
            let mut is_prime = true;
            for &p in cache.iter() {
                if p > candidate / p {
                    break;
                }
                if candidate.is_multiple_of(p) {
                    is_prime = false;
                    break;
                }
            }
            if is_prime {
                cache.push(candidate);
            }
            candidate += 2;
        }
    }
    cache[index]
}

/// Radical inverse of `index` in the given integer `base`.
///
/// Writes `index` in base `base` and mirrors the digits about the radix
/// point, giving a value in `[0, 1)`. This is the van der Corput sequence,
/// the one-dimensional building block of Halton point sets.
///
/// # Panics
///
/// Panics when `base < 2`.
///
/// # Examples
///
/// ```
/// use eindir_core::radical_inverse;
///
/// assert_eq!(radical_inverse(0, 2), 0.0);
/// assert_eq!(radical_inverse(1, 2), 0.5);
/// assert_eq!(radical_inverse(3, 2), 0.75);
/// assert_eq!(radical_inverse(1, 3), 1.0 / 3.0);
/// ```
pub fn radical_inverse(mut index: u64, base: u64) -> f64 {
    assert!(base >= 2, "radical inverse base must be at least 2");
    let inv_base = 1.0 / base as f64;
    let mut fraction = inv_base;
    let mut value = 0.0;
    while index > 0 {
        value += (index % base) as f64 * fraction;
        index /= base;
        fraction *= inv_base;
    }
    value
}

/// The Halton point of number `index` in the unit hypercube `[0, 1)^dim`.
///
/// Axis `i` is the radical inverse of `index` in the `i`-th prime
/// (2, 3, 5, ...). Index 0 is the origin. Primes beyond the first 32 are
/// computed once and cached for the life of the process.
///
/// # Examples
///
/// ```
/// use eindir_core::halton_unit;
/// use ndarray::array;
///
/// assert_eq!(halton_unit(1, 2), array![0.5, 1.0 / 3.0]);
/// ```
pub fn halton_unit(index: u64, dim: usize) -> Array1<f64> {
    Array1::from_iter((0..dim).map(|axis| radical_inverse(index, nth_prime(axis))))
}

/// A Halton point set scaled to the supplied box bounds.
///
/// Returns an `n x bounds.dims` array whose row `r` is
/// `low + (high - low) * halton_unit(skip + r, dims)`. `skip` is the first
/// Halton index; `skip = 1` avoids the corner `low`. `bounds.slack` is not
/// used. Designs of 64 rows or more fill rows in parallel with Rayon, and
/// the result does not depend on the thread count.
///
/// # Panics
///
/// Panics when `high[i] < low[i]` on any axis, which [`Bounds::new`]
/// already refuses.
///
/// # Examples
///
/// ```
/// use eindir_core::{Bounds, halton_points};
/// use ndarray::array;
///
/// let b = Bounds::new(array![0.0, -1.0], array![2.0, 1.0], 0.0);
/// let pts = halton_points(&b, 3, 1);
/// assert_eq!(pts.shape(), &[3, 2]);
/// assert_eq!(pts.row(0), array![1.0, -1.0 + 2.0 / 3.0]);
/// ```
pub fn halton_points(bounds: &Bounds<f64>, n: usize, skip: u64) -> Array2<f64> {
    let dim = bounds.dims;
    if dim > 0 {
        // Serial prime warm-up so parallel rows only read the cache.
        let _ = nth_prime(dim - 1);
    }
    let widths: Vec<f64> = (0..dim)
        .map(|axis| {
            let width = bounds.high[axis] - bounds.low[axis];
            assert!(width >= 0.0, "bounds require high >= low on every axis");
            width
        })
        .collect();
    let lows = bounds
        .low
        .as_slice()
        .expect("bounds.low contiguous")
        .to_vec();

    let fill_row = |row: usize| -> Vec<f64> {
        let unit = halton_unit(skip + row as u64, dim);
        (0..dim)
            .map(|axis| lows[axis] + widths[axis] * unit[axis])
            .collect()
    };
    let flat: Vec<f64> = if n < HALTON_PARALLEL_MIN {
        (0..n).flat_map(fill_row).collect()
    } else {
        (0..n).into_par_iter().flat_map(fill_row).collect()
    };
    Array2::from_shape_vec((n, dim), flat).expect("halton shape")
}

/// Default bounded low-discrepancy design used by optimization front-ends.
///
/// Currently the same as [`halton_points`]; front-ends call this name so
/// the default design can change without touching them. The Python
/// function of the same name calls it.
///
/// # Panics
///
/// As [`halton_points`].
///
/// # Examples
///
/// ```
/// use eindir_core::{Bounds, halton_points, low_discrepancy_points};
/// use ndarray::array;
///
/// let b = Bounds::new(array![-5.12, -5.12], array![5.12, 5.12], 1e-6);
/// assert_eq!(low_discrepancy_points(&b, 8, 1), halton_points(&b, 8, 1));
/// ```
pub fn low_discrepancy_points(bounds: &Bounds<f64>, n: usize, skip: u64) -> Array2<f64> {
    halton_points(bounds, n, skip)
}

/// Shifted Halton point set scaled to the supplied box bounds.
///
/// The shift is applied in the unit hypercube modulo one, then scaled to the
/// target box. This gives a deterministic replicated design for randomized-QMC
/// style restarts while preserving bounded points. `shift` holds one value
/// per axis, normally in `[0, 1)`.
///
/// # Panics
///
/// Panics when `shift.len() != bounds.dims`, and as [`halton_points`].
pub fn shifted_halton_points(
    bounds: &Bounds<f64>,
    n: usize,
    skip: u64,
    shift: ArrayView1<'_, f64>,
) -> Array2<f64> {
    assert_eq!(
        shift.len(),
        bounds.dims,
        "shift dimension must match bounds dimension"
    );
    let dim = bounds.dims;
    if dim > 0 {
        let _ = nth_prime(dim - 1);
    }
    let widths: Vec<f64> = (0..dim)
        .map(|axis| {
            let width = bounds.high[axis] - bounds.low[axis];
            assert!(width >= 0.0, "bounds require high >= low on every axis");
            width
        })
        .collect();
    let lows = bounds
        .low
        .as_slice()
        .expect("bounds.low contiguous")
        .to_vec();
    let shift_v = shift.to_vec();

    let fill_row = |row: usize| -> Vec<f64> {
        let unit = halton_unit(skip + row as u64, dim);
        (0..dim)
            .map(|axis| {
                let shifted = (unit[axis] + shift_v[axis]).fract();
                lows[axis] + widths[axis] * shifted
            })
            .collect()
    };
    let flat: Vec<f64> = if n < HALTON_PARALLEL_MIN {
        (0..n).flat_map(fill_row).collect()
    } else {
        (0..n).into_par_iter().flat_map(fill_row).collect()
    };
    Array2::from_shape_vec((n, dim), flat).expect("shifted halton shape")
}

/// Deterministically shifted bounded low-discrepancy design.
///
/// The supplied seed chooses a reproducible Cranley-Patterson style shift in
/// the unit hypercube. Different seeds give independent replicas of the same
/// underlying low-discrepancy design. The shift comes from
/// `rand::rngs::StdRng`, so the points for a given seed can change when the
/// `rand` crate changes its generator.
///
/// # Panics
///
/// As [`halton_points`].
pub fn shifted_low_discrepancy_points(
    bounds: &Bounds<f64>,
    n: usize,
    skip: u64,
    seed: u64,
) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let shift = Array1::from_iter((0..bounds.dims).map(|_| rng.random::<f64>()));
    shifted_halton_points(bounds, n, skip, shift.view())
}

fn vertex_count_for_design(dim: usize, n: usize) -> Option<usize> {
    if dim >= usize::BITS as usize {
        return None;
    }
    let count = 1usize << dim;
    (count <= n).then_some(count)
}

/// Boundary-anchored bounded low-discrepancy design.
///
/// Starts from [`halton_points`] and overwrites the first rows:
///
/// - when `2^dims <= n`, rows `0..2^dims` become the box vertices and, if a
///   row remains, row `2^dims` becomes the box centre;
/// - otherwise row 0 becomes `low`, row 1 `high` and row 2 the centre, as
///   far as `n` allows.
///
/// The other rows keep their Halton values.
///
/// # Panics
///
/// As [`halton_points`].
///
/// # Examples
///
/// ```
/// use eindir_core::{Bounds, boundary_anchored_low_discrepancy_points};
/// use ndarray::array;
///
/// let b = Bounds::new(array![0.0, 0.0], array![1.0, 1.0], 0.0);
/// let pts = boundary_anchored_low_discrepancy_points(&b, 6, 1);
/// assert_eq!(pts.row(0), array![0.0, 0.0]);
/// assert_eq!(pts.row(3), array![1.0, 1.0]);
/// assert_eq!(pts.row(4), array![0.5, 0.5]);
/// ```
pub fn boundary_anchored_low_discrepancy_points(
    bounds: &Bounds<f64>,
    n: usize,
    skip: u64,
) -> Array2<f64> {
    let mut out = halton_points(bounds, n, skip);
    if n == 0 {
        return out;
    }
    if let Some(vertex_count) = vertex_count_for_design(bounds.dims, n) {
        for row in 0..vertex_count {
            for axis in 0..bounds.dims {
                out[[row, axis]] = if ((row >> axis) & 1) == 0 {
                    bounds.low[axis]
                } else {
                    bounds.high[axis]
                };
            }
        }
        if vertex_count < n {
            for axis in 0..bounds.dims {
                out[[vertex_count, axis]] = 0.5 * (bounds.low[axis] + bounds.high[axis]);
            }
        }
        return out;
    }
    for axis in 0..bounds.dims {
        out[[0, axis]] = bounds.low[axis];
    }
    if n > 1 {
        for axis in 0..bounds.dims {
            out[[1, axis]] = bounds.high[axis];
        }
    }
    if n > 2 {
        for axis in 0..bounds.dims {
            out[[2, axis]] = 0.5 * (bounds.low[axis] + bounds.high[axis]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nth_prime_matches_known_values() {
        // Small primes table and the first few beyond it.
        assert_eq!(nth_prime(0), 2);
        assert_eq!(nth_prime(31), 131);
        assert_eq!(nth_prime(32), 137);
        // The 1000th prime (0-indexed 999) is 7919.
        assert_eq!(nth_prime(999), 7919);
        // The 10000th prime (0-indexed 9999) is 104729.
        assert_eq!(nth_prime(9999), 104729);
    }

    #[test]
    fn high_dimensional_halton_is_fast_and_bounded() {
        // A 34k-dimensional design must build the prime table once and stay
        // in the unit cube; the per-call trial-division version stalled here.
        let unit = halton_unit(7, 34_134);
        assert_eq!(unit.len(), 34_134);
        assert!(unit.iter().all(|&u| (0.0..1.0).contains(&u)));
    }
}
