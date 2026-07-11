//! Low-discrepancy point sets for bounded numerical objectives.

use std::sync::{Mutex, OnceLock};

use ndarray::{Array1, Array2, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::Bounds;

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
/// This is the one-dimensional building block of Halton point sets.
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

/// A single `dim`-dimensional Halton point in the unit hypercube.
pub fn halton_unit(index: u64, dim: usize) -> Array1<f64> {
    Array1::from_iter((0..dim).map(|axis| radical_inverse(index, nth_prime(axis))))
}

/// A Halton point set scaled to the supplied box bounds.
///
/// `skip` controls the first Halton index. Use `skip = 1` to avoid the origin.
pub fn halton_points(bounds: &Bounds<f64>, n: usize, skip: u64) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((n, bounds.dims));
    for row in 0..n {
        let unit = halton_unit(skip + row as u64, bounds.dims);
        for axis in 0..bounds.dims {
            let width = bounds.high[axis] - bounds.low[axis];
            assert!(width >= 0.0, "bounds require high >= low on every axis");
            out[[row, axis]] = bounds.low[axis] + width * unit[axis];
        }
    }
    out
}

/// Default bounded low-discrepancy design used by optimization front-ends.
pub fn low_discrepancy_points(bounds: &Bounds<f64>, n: usize, skip: u64) -> Array2<f64> {
    halton_points(bounds, n, skip)
}

/// Shifted Halton point set scaled to the supplied box bounds.
///
/// The shift is applied in the unit hypercube modulo one, then scaled to the
/// target box. This gives a deterministic replicated design for randomized-QMC
/// style restarts while preserving bounded points.
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
    let mut out = Array2::<f64>::zeros((n, bounds.dims));
    for row in 0..n {
        let unit = halton_unit(skip + row as u64, bounds.dims);
        for axis in 0..bounds.dims {
            let width = bounds.high[axis] - bounds.low[axis];
            assert!(width >= 0.0, "bounds require high >= low on every axis");
            let shifted = (unit[axis] + shift[axis]).fract();
            out[[row, axis]] = bounds.low[axis] + width * shifted;
        }
    }
    out
}

/// Deterministically shifted bounded low-discrepancy design.
///
/// The supplied seed chooses a reproducible Cranley-Patterson style shift in
/// the unit hypercube. Different seeds give independent replicas of the same
/// underlying low-discrepancy design.
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
/// The design starts with all box vertices when the requested size can hold
/// them. For larger boxes it anchors the two diagonal vertices and keeps the
/// remaining rows as the Halton design.
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
