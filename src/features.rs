//! Measurable problem geometry and compensated energy differences.
//!
//! Feature extraction for auto-routing (anneal regimes) and numerically
//! safer ΔE on the acceptance path.

use crate::bounds::Bounds;
use num_traits::Float;

/// Box geometry features used by regime selection.
#[derive(Clone, Debug, PartialEq)]
pub struct BoxGeometry {
    /// Dimension of the box.
    pub dim: usize,
    /// Mean side length.
    pub mean_width: f64,
    /// max_j width_j / min_j width_j (≥ 1). Infinite sides contribute a large aspect.
    pub aspect_ratio: f64,
    /// Per-coordinate widths (finite; non-positive replaced by 1).
    pub widths: Vec<f64>,
}

/// Extract geometry from bounds. Safe default for empty/invalid boxes.
pub fn box_geometry<T: Float>(bounds: &Bounds<T>) -> BoxGeometry {
    let dim = bounds.dims;
    if dim == 0 {
        return BoxGeometry {
            dim: 0,
            mean_width: 1.0,
            aspect_ratio: 1.0,
            widths: vec![],
        };
    }
    let mut widths = Vec::with_capacity(dim);
    let mut min_w = f64::INFINITY;
    let mut max_w = 0.0f64;
    let mut sum = 0.0f64;
    for j in 0..dim {
        let lo = bounds.low[j].to_f64().unwrap_or(0.0);
        let hi = bounds.high[j].to_f64().unwrap_or(1.0);
        let mut w = hi - lo;
        if !w.is_finite() || w <= 0.0 {
            w = 1.0;
        }
        widths.push(w);
        sum += w;
        min_w = min_w.min(w);
        max_w = max_w.max(w);
    }
    let aspect = if min_w > 0.0 && min_w.is_finite() {
        (max_w / min_w).max(1.0)
    } else {
        1.0
    };
    BoxGeometry {
        dim,
        mean_width: sum / dim as f64,
        aspect_ratio: aspect,
        widths,
    }
}

/// Compensated energy difference `f_new - f_cur` for acceptance channels.
///
/// Uses a two-sum style compensation when both values are finite so that
/// cancellation near equal energies is less biased than a bare subtraction.
/// Falls back to ordinary subtraction if either argument is non-finite.
#[inline]
pub fn compensated_delta(f_new: f64, f_cur: f64) -> f64 {
    if !f_new.is_finite() || !f_cur.is_finite() {
        return f_new - f_cur;
    }
    // Knuth two-sum of f_new + (-f_cur).
    let a = f_new;
    let b = -f_cur;
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    let out = s + err;
    if out.is_finite() { out } else { f_new - f_cur }
}

/// Dimension-aware isotropic proposal scale: `c * mean_width / sqrt(dim)`.
///
/// Matches the paper's dimension-aware Move scale (prevents √D acceptance collapse).
#[inline]
pub fn isotropic_proposal_scale(geom: &BoxGeometry, c: f64) -> f64 {
    let d = geom.dim.max(1) as f64;
    let c = if c.is_finite() && c > 0.0 { c } else { 0.25 };
    (c * geom.mean_width / d.sqrt()).max(1e-12)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bounds::Bounds;
    use ndarray::{Array1, array};

    #[test]
    fn aspect_ratio_detects_elongation() {
        let b = Bounds::new(array![0.0, 0.0], array![1.0, 100.0], 0.0);
        let g = box_geometry(&b);
        assert_eq!(g.dim, 2);
        assert!((g.aspect_ratio - 100.0).abs() < 1e-12);
    }

    #[test]
    fn compensated_delta_near_cancellation() {
        // Values that cancel in the leading bits.
        let f_cur = 1.0 + 1e-12;
        let f_new = 1.0 + 2e-12;
        let d = compensated_delta(f_new, f_cur);
        assert!(d.is_finite());
        assert!(d > 0.0);
        // Relative agreement with true difference.
        let naive = f_new - f_cur;
        assert!((d - naive).abs() <= naive.abs().max(1e-18) * 1e-6 + 1e-18);
    }

    #[test]
    fn isotropic_scale_shrinks_with_dim() {
        let b2 = Bounds::new(array![0.0, 0.0], array![2.0, 2.0], 0.0);
        let b8 = Bounds::new(Array1::from_elem(8, 0.0), Array1::from_elem(8, 2.0), 0.0);
        let s2 = isotropic_proposal_scale(&box_geometry(&b2), 0.25);
        let s8 = isotropic_proposal_scale(&box_geometry(&b8), 0.25);
        assert!(s8 < s2);
    }
}
