//! Dimension-collapse and Chebyshev surrogate components for the typed algebra.
//!
//! These two primitives are `Obj`-transforms: each takes the objective slot of
//! the typed component algebra and returns another objective, so any sampler
//! built on the algebra (single-chain SA, the MCMC variants, parallel
//! tempering, or the HMC point) consumes them through the same `Objective`
//! trait without modification. A high-dimensional objective is made tractable
//! by collapsing the search to a low-dimensional active subspace
//! (`ReducedObjective`) and, after a pilot phase, replacing the restricted
//! objective by a cheap total-degree Chebyshev model with an analytic gradient
//! (`ChebyshevSurrogate`). Because both are objectives, the same collapse and
//! surrogate serve every point of the algebra at once.

use crate::{Bounds, Objective, gradient::Gradient};
use ndarray::{Array1, Array2, ArrayView1};

/// Affine dimension-collapse of an inner objective onto a `k`-dimensional box.
///
/// Given a full-space origin `x0 in R^n` and a basis `W in R^{n x k}` whose
/// columns span the retained subspace (for example the dominant eigenvectors
/// of the pilot gradient covariance, an active subspace), the reduced
/// coordinate `r in R^k` decodes to `x = clip(x0 + W r)` in the inner box. The
/// reduced objective evaluates the inner objective at the decoded point, so a
/// sampler searches in `R^k` while every value is the true objective.
pub struct ReducedObjective<O: Objective<f64>> {
    inner: O,
    origin: Array1<f64>,
    basis: Array2<f64>,
    bounds: Bounds<f64>,
}

impl<O: Objective<f64>> ReducedObjective<O> {
    /// Builds a reduced objective from an inner objective, a full-space origin,
    /// an `n x k` basis, and the reduced box `bounds` in `R^k`.
    ///
    /// Panics when the shapes disagree: `origin.len()` and `basis.nrows()` must
    /// equal `inner.dim()`, and `basis.ncols()` must equal `bounds.dims`.
    pub fn new(inner: O, origin: Array1<f64>, basis: Array2<f64>, bounds: Bounds<f64>) -> Self {
        assert_eq!(
            origin.len(),
            inner.dim(),
            "origin length must match inner dim"
        );
        assert_eq!(
            basis.nrows(),
            inner.dim(),
            "basis rows must match inner dim"
        );
        assert_eq!(
            basis.ncols(),
            bounds.dims,
            "basis cols must match reduced dim"
        );
        Self {
            inner,
            origin,
            basis,
            bounds,
        }
    }

    /// Decodes a reduced coordinate `r in R^k` to a full point `x in R^n`,
    /// clipped to the inner objective's box.
    pub fn decode(&self, r: ArrayView1<f64>) -> Array1<f64> {
        let lifted = &self.origin + self.basis.dot(&r);
        self.inner.bounds().clip(lifted.view())
    }

    /// Borrows the wrapped full-dimensional objective.
    pub fn inner(&self) -> &O {
        &self.inner
    }
}

impl<O: Objective<f64>> Objective<f64> for ReducedObjective<O> {
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, r: ArrayView1<f64>) -> f64 {
        self.inner.eval(self.decode(r).view())
    }
}

/// Total-degree Chebyshev surrogate objective on a box in `R^k`.
///
/// The surrogate stores a list of multi-indices `terms` (each a length-`k`
/// degree vector with total degree at most the fit degree) and a matching
/// coefficient per term. A point `x` in the box `[low, high]` maps coordinate
/// by coordinate to `t in [-1, 1]^k`; the surrogate value is
/// `sum_j coeff_j * prod_d T_{terms_j[d]}(t_d)`, where `T_n` is the Chebyshev
/// polynomial of the first kind. Coefficients are fit elsewhere (least squares
/// on pilot samples in the reduced box); this type supplies the cheap value
/// and the analytic gradient that the HMC point consumes.
pub struct ChebyshevSurrogate {
    bounds: Bounds<f64>,
    terms: Vec<Vec<usize>>,
    coeffs: Array1<f64>,
}

impl ChebyshevSurrogate {
    /// Builds a surrogate from a reduced box, the total-degree multi-index set,
    /// and one coefficient per multi-index.
    ///
    /// Panics when `terms.len() != coeffs.len()` or when any multi-index length
    /// differs from `bounds.dims`.
    pub fn new(bounds: Bounds<f64>, terms: Vec<Vec<usize>>, coeffs: Array1<f64>) -> Self {
        assert_eq!(
            terms.len(),
            coeffs.len(),
            "one coefficient per term required"
        );
        assert!(
            terms.iter().all(|t| t.len() == bounds.dims),
            "every multi-index must have length bounds.dims",
        );
        Self {
            bounds,
            terms,
            coeffs,
        }
    }

    /// Maps a box coordinate to `[-1, 1]`; degenerate intervals map to zero.
    fn to_unit(&self, d: usize, xd: f64) -> f64 {
        let lo = self.bounds.low[d];
        let hi = self.bounds.high[d];
        let span = hi - lo;
        if span.abs() <= f64::EPSILON {
            0.0
        } else {
            (2.0 * (xd - lo) / span - 1.0).clamp(-1.0, 1.0)
        }
    }

    /// Chebyshev value and `d/dt` derivative up to `n` by the standard
    /// recurrences `T_{m+1} = 2 t T_m - T_{m-1}` and `T'_{m+1} = 2 T_m + 2 t
    /// T'_m - T'_{m-1}`.
    fn cheb_with_deriv(n: usize, t: f64) -> (Vec<f64>, Vec<f64>) {
        let mut vals = vec![0.0; n + 1];
        let mut ders = vec![0.0; n + 1];
        vals[0] = 1.0;
        ders[0] = 0.0;
        if n >= 1 {
            vals[1] = t;
            ders[1] = 1.0;
        }
        for m in 1..n {
            vals[m + 1] = 2.0 * t * vals[m] - vals[m - 1];
            ders[m + 1] = 2.0 * vals[m] + 2.0 * t * ders[m] - ders[m - 1];
        }
        (vals, ders)
    }

    /// Per-dimension Chebyshev value/derivative tables for a point `x`.
    fn tables(&self, x: ArrayView1<f64>) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let k = self.bounds.dims;
        let max_deg: Vec<usize> = (0..k)
            .map(|d| self.terms.iter().map(|t| t[d]).max().unwrap_or(0))
            .collect();
        let mut vals = Vec::with_capacity(k);
        let mut ders = Vec::with_capacity(k);
        for d in 0..k {
            let (v, dv) = Self::cheb_with_deriv(max_deg[d], self.to_unit(d, x[d]));
            vals.push(v);
            ders.push(dv);
        }
        (vals, ders)
    }

    /// Analytic gradient of the surrogate at `x`, in box coordinates.
    pub fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let k = self.bounds.dims;
        let (vals, ders) = self.tables(x);
        let mut g = Array1::<f64>::zeros(k);
        for (term, &c) in self.terms.iter().zip(self.coeffs.iter()) {
            for j in 0..k {
                let span = self.bounds.high[j] - self.bounds.low[j];
                if span.abs() <= f64::EPSILON {
                    continue;
                }
                let dt_dx = 2.0 / span;
                let mut prod = ders[j][term[j]] * dt_dx;
                for (d, item) in term.iter().enumerate() {
                    if d != j {
                        prod *= vals[d][*item];
                    }
                }
                g[j] += c * prod;
            }
        }
        g
    }
}

impl Objective<f64> for ChebyshevSurrogate {
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let (vals, _) = self.tables(x);
        let mut acc = 0.0;
        for (term, &c) in self.terms.iter().zip(self.coeffs.iter()) {
            let mut prod = c;
            for (d, &deg) in term.iter().enumerate() {
                prod *= vals[d][deg];
            }
            acc += prod;
        }
        acc
    }
}

impl Gradient<f64> for ChebyshevSurrogate {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        // Delegate to the existing analytic implementation (Ceres-style
        // native gradient).  The inherent `grad` remains for back-compat.
        ChebyshevSurrogate::grad(self, x)
    }
    fn dim(&self) -> usize {
        self.bounds.dims
    }
}

impl<O: Objective<f64> + Gradient<f64>> Gradient<f64> for ReducedObjective<O> {
    fn grad(&self, r: ArrayView1<f64>) -> Array1<f64> {
        // Chain rule through the affine decoder: g_r = W^T @ g_full(decode(r))
        let x = self.decode(r);
        let g_full = self.inner.grad(x.view());
        self.basis.t().dot(&g_full)
    }
    fn dim(&self) -> usize {
        self.bounds.dims
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objectives::StybTang2D;
    use ndarray::{Array1, Array2, array};

    fn reduced_box(k: usize, lo: f64, hi: f64) -> Bounds<f64> {
        Bounds::new(Array1::from_elem(k, lo), Array1::from_elem(k, hi), 1e-9)
    }

    #[test]
    fn reduced_decodes_and_evaluates_inner() {
        // Collapse the 2D Styblinski-Tang onto the x1 = x2 diagonal: a single
        // reduced coordinate r maps to (r, r).
        let inner = StybTang2D::new();
        let origin = array![0.0, 0.0];
        let basis = Array2::from_shape_vec((2, 1), vec![1.0, 1.0]).unwrap();
        let reduced = ReducedObjective::new(inner, origin, basis, reduced_box(1, -5.0, 5.0));
        assert_eq!(Objective::dim(&reduced), 1);
        // Decode is the diagonal lift.
        let x = reduced.decode(array![2.0].view());
        assert!((x[0] - 2.0).abs() < 1e-12 && (x[1] - 2.0).abs() < 1e-12);
        // Value equals the inner objective on the diagonal.
        let direct = StybTang2D::new().eval(array![2.0, 2.0].view());
        assert!((reduced.eval(array![2.0].view()) - direct).abs() < 1e-12);
    }

    #[test]
    fn chebyshev_reproduces_a_known_polynomial() {
        // f(x) = 3 + 2*T_1(t) on [-1, 1] equals 3 + 2x; check value and slope.
        let terms = vec![vec![0usize], vec![1usize]];
        let coeffs = array![3.0, 2.0];
        let s = ChebyshevSurrogate::new(reduced_box(1, -1.0, 1.0), terms, coeffs);
        assert!((s.eval(array![0.5].view()) - 4.0).abs() < 1e-12);
        assert!((s.grad(array![0.5].view())[0] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn chebyshev_gradient_matches_finite_difference() {
        // A two-dimensional mixed-degree surrogate; compare the analytic
        // gradient to a central finite difference.
        let terms = vec![vec![2, 0], vec![1, 1], vec![0, 2]];
        let coeffs = array![0.7, -1.3, 0.4];
        let s = ChebyshevSurrogate::new(reduced_box(2, -2.0, 3.0), terms, coeffs);
        let x = array![0.6, 1.1];
        let g = s.grad(x.view());
        let h = 1e-6;
        for j in 0..2 {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[j] += h;
            xm[j] -= h;
            let fd = (s.eval(xp.view()) - s.eval(xm.view())) / (2.0 * h);
            assert!(
                (g[j] - fd).abs() < 1e-5,
                "dim {j}: analytic {} vs fd {}",
                g[j],
                fd
            );
        }
    }
}
