//! Low-discrepancy point sets for bounded numerical objectives.

use ndarray::{Array1, Array2};

use crate::Bounds;

const SMALL_PRIMES: [u64; 32] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131,
];

fn is_prime(candidate: u64) -> bool {
    if candidate < 2 {
        return false;
    }
    if candidate == 2 {
        return true;
    }
    if candidate % 2 == 0 {
        return false;
    }
    let mut divisor = 3;
    while divisor * divisor <= candidate {
        if candidate % divisor == 0 {
            return false;
        }
        divisor += 2;
    }
    true
}

fn nth_prime(index: usize) -> u64 {
    if index < SMALL_PRIMES.len() {
        return SMALL_PRIMES[index];
    }
    let mut count = SMALL_PRIMES.len();
    let mut candidate = SMALL_PRIMES[SMALL_PRIMES.len() - 1] + 2;
    loop {
        if is_prime(candidate) {
            if count == index {
                return candidate;
            }
            count += 1;
        }
        candidate += 2;
    }
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
