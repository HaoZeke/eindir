//! Separable rank-1 surrogate: the base case of the functional tensor train.
//!
//! A high-dimensional objective is modelled additively,
//! `f(x) ~= c + sum_j g_j(x_j)`, with each `g_j` a first-kind Chebyshev energy
//! in one coordinate. This is a rank-1 functional tensor train in the *full*
//! `d` coordinates, so it carries no active-subspace collapse and the tempered
//! Gibbs density `exp(-f/T)` factorises across coordinates exactly when `f` is
//! separable. That factorisation is what the [`AdditiveSurrogate::sample`]
//! independence proposal exploits: each coordinate is drawn from its own 1D
//! tempered marginal `exp(-g_j/T)/Z_j(T)` by inverse-CDF, in `O(d m)` with no
//! curse of dimensionality, so a single draw places every coordinate at its
//! tempered optimum at once -- the regime an active-subspace surrogate cannot
//! reach.
//!
//! Like [`crate::reduced::ChebyshevSurrogate`], the surrogate is an
//! `Obj`-transform: it implements [`Objective`], so every point of the sampling
//! algebra consumes it through the same trait. The fit decouples per coordinate
//! (one `degree x degree` solve each), which keeps it linear in `d` and usable
//! at native CUTEst sizes. A Metropolis accept against the true objective
//! debiases the mean-field error when the objective is not separable.

use crate::{Bounds, Objective, gradient::Gradient};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::Rng;

/// First-kind Chebyshev values `T_1..T_degree` at a unit coordinate `t` in
/// `[-1, 1]` (the constant `T_0` is carried by a shared intercept and omitted).
fn cheb_basis(t: f64, degree: usize) -> Array1<f64> {
    let mut out = Array1::zeros(degree);
    if degree == 0 {
        return out;
    }
    let mut tm1 = 1.0; // T_0
    let mut tm = t; // T_1
    out[0] = tm;
    for m in 1..degree {
        let next = 2.0 * t * tm - tm1;
        out[m] = next;
        tm1 = tm;
        tm = next;
    }
    out
}

/// Derivatives `T_1'..T_degree'` at unit coordinate `t`, via the coupled
/// recurrence `T'_{m+1} = 2 T_m + 2 t T'_m - T'_{m-1}`.
fn cheb_basis_deriv(t: f64, degree: usize) -> Array1<f64> {
    let mut dv = Array1::zeros(degree);
    if degree == 0 {
        return dv;
    }
    // values needed for the derivative recurrence
    let mut v_m1 = 1.0; // T_0
    let mut v_m = t; // T_1
    let mut d_m1 = 0.0; // T_0'
    let mut d_m = 1.0; // T_1'
    dv[0] = d_m;
    for m in 1..degree {
        let d_next = 2.0 * v_m + 2.0 * t * d_m - d_m1;
        dv[m] = d_next;
        let v_next = 2.0 * t * v_m - v_m1;
        v_m1 = v_m;
        v_m = v_next;
        d_m1 = d_m;
        d_m = d_next;
    }
    dv
}

/// Cholesky factor `L` (lower) of a symmetric positive-(semi)definite `a` with a
/// small Tikhonov floor; `a` is `p x p` with `p = degree` tiny. The ridge keeps
/// the normal equations solvable when a coordinate's pilot design is
/// rank-deficient (e.g. a near-constant column).
fn cholesky_ridged(mut a: Array2<f64>) -> Array2<f64> {
    let p = a.nrows();
    let ridge = 1e-10 * (1.0 + (0..p).map(|i| a[[i, i]]).fold(0.0, f64::max));
    for i in 0..p {
        a[[i, i]] += ridge;
    }
    let mut l = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                l[[i, j]] = s.max(ridge).sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    l
}

/// Solve `a x = b` given the Cholesky factor `l` of `a` (`a = L L^T`).
fn chol_solve(l: &Array2<f64>, b: Array1<f64>) -> Array1<f64> {
    let p = l.nrows();
    // forward solve L y = b
    let mut y = Array1::<f64>::zeros(p);
    for i in 0..p {
        let mut s = b[i];
        for k in 0..i {
            s -= l[[i, k]] * y[k];
        }
        y[i] = s / l[[i, i]];
    }
    // back solve L^T x = y
    let mut x = Array1::<f64>::zeros(p);
    for i in (0..p).rev() {
        let mut s = y[i];
        for k in (i + 1)..p {
            s -= l[[k, i]] * x[k];
        }
        x[i] = s / l[[i, i]];
    }
    x
}

/// A separable rank-1 Chebyshev energy model `c + sum_j g_j(x_j)`.
pub struct AdditiveSurrogate {
    bounds: Bounds<f64>,
    intercept: f64,
    /// `(dim, degree)` per-coordinate Chebyshev coefficients for `T_1..T_degree`.
    coeffs: Array2<f64>,
    /// `(dim, degree)` pilot column means of each coordinate's basis. Under
    /// uniform sampling the even-degree Chebyshev columns have nonzero mean
    /// (`E[T_2] = -1/3`, ...), so the per-coordinate fit is consistent only on
    /// mean-centred columns; the offset is folded into `intercept` and
    /// subtracted back in [`value`](Self::value).
    col_means: Array2<f64>,
    degree: usize,
}

impl AdditiveSurrogate {
    /// Fit the surrogate to pilot points `x` (`n x d`) and values `y` (`n`) with
    /// a default of 12 backfitting passes.
    pub fn fit(x: ArrayView2<f64>, y: ArrayView1<f64>, bounds: Bounds<f64>, degree: usize) -> Self {
        Self::fit_backfit(x, y, bounds, degree, 12)
    }

    /// Fit by backfitting (Gauss-Seidel over coordinates).
    ///
    /// The intercept is the pilot mean. Each coordinate's Chebyshev coefficients
    /// are refit in turn against the *partial* residual `y - intercept -
    /// sum_{l != j} g_l`, on a mean-centred design so the fit is consistent
    /// under uniform sampling. Sweeping all coordinates `n_passes` times removes
    /// the finite-sample cross-coordinate contamination that an independent
    /// per-coordinate fit leaves behind, converging to the joint least-squares
    /// additive model while staying `O(n_passes d (n degree + degree^3))` --
    /// linear in `d`, unlike the joint solve. Each coordinate's Gram factor is
    /// cached, so a pass costs only one triangular solve per coordinate.
    ///
    /// Panics when shapes disagree or `degree == 0`.
    pub fn fit_backfit(
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        bounds: Bounds<f64>,
        degree: usize,
        n_passes: usize,
    ) -> Self {
        assert!(degree >= 1, "degree must be at least 1");
        let n = x.nrows();
        let dim = x.ncols();
        assert_eq!(n, y.len(), "x rows must match y length");
        assert_eq!(dim, bounds.dims, "x cols must match bounds dims");
        assert!(n >= degree + 1, "need at least degree+1 pilot points");

        let mean_y = y.sum() / n as f64;
        let mut col_means = Array2::<f64>::zeros((dim, degree));
        let low = &bounds.low;
        let high = &bounds.high;

        // Precompute each coordinate's centred design B_j and the Cholesky
        // factor of its Gram; only the right-hand side changes between passes.
        let mut designs: Vec<Array2<f64>> = Vec::with_capacity(dim);
        let mut chols: Vec<Array2<f64>> = Vec::with_capacity(dim);
        for j in 0..dim {
            let span = {
                let w = high[j] - low[j];
                if w > 0.0 { w } else { 1.0 }
            };
            let mut d = Array2::<f64>::zeros((n, degree));
            for i in 0..n {
                let t = 2.0 * (x[[i, j]] - low[j]) / span - 1.0;
                d.row_mut(i).assign(&cheb_basis(t.clamp(-1.0, 1.0), degree));
            }
            let means: Array1<f64> = d.mean_axis(ndarray::Axis(0)).unwrap();
            col_means.row_mut(j).assign(&means);
            for mut row in d.rows_mut() {
                row -= &means;
            }
            chols.push(cholesky_ridged(d.t().dot(&d)));
            designs.push(d);
        }

        // Backfitting: maintain each coordinate's fitted contribution over the
        // pilot and the running total, so the partial residual is one subtraction.
        let target: Array1<f64> = y.to_owned() - mean_y;
        let mut coeffs = Array2::<f64>::zeros((dim, degree));
        let mut fitted: Vec<Array1<f64>> = vec![Array1::zeros(n); dim];
        let mut total: Array1<f64> = Array1::zeros(n);
        for _ in 0..n_passes.max(1) {
            for j in 0..dim {
                let partial = &target - &(&total - &fitted[j]);
                let rhs = designs[j].t().dot(&partial);
                let c = chol_solve(&chols[j], rhs);
                let new_fit = designs[j].dot(&c);
                total = &total - &fitted[j] + &new_fit;
                fitted[j] = new_fit;
                coeffs.row_mut(j).assign(&c);
            }
        }
        Self {
            bounds,
            intercept: mean_y,
            coeffs,
            col_means,
            degree,
        }
    }

    /// Search-space box of the surrogate.
    pub fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn unit(&self, j: usize, xj: f64) -> f64 {
        let (lo, hi) = (self.bounds.low[j], self.bounds.high[j]);
        let span = if hi > lo { hi - lo } else { 1.0 };
        (2.0 * (xj - lo) / span - 1.0).clamp(-1.0, 1.0)
    }

    /// Energy contribution `g_j(xj)` of one coordinate (no intercept), centred
    /// by the pilot column means so the per-coordinate fit is consistent.
    fn coord_energy(&self, j: usize, xj: f64) -> f64 {
        let t = self.unit(j, xj);
        let basis = cheb_basis(t, self.degree);
        (&basis - &self.col_means.row(j)).dot(&self.coeffs.row(j))
    }

    /// Surrogate value at full point `x`.
    pub fn value(&self, x: ArrayView1<f64>) -> f64 {
        let mut acc = self.intercept;
        for j in 0..self.bounds.dims {
            acc += self.coord_energy(j, x[j]);
        }
        acc
    }

    /// Analytic gradient of the separable surrogate at `x`.
    pub fn gradient(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let dim = self.bounds.dims;
        let mut g = Array1::<f64>::zeros(dim);
        for j in 0..dim {
            let (lo, hi) = (self.bounds.low[j], self.bounds.high[j]);
            let span = if hi > lo { hi - lo } else { 1.0 };
            let t = self.unit(j, x[j]);
            let dgdt = cheb_basis_deriv(t, self.degree).dot(&self.coeffs.row(j));
            g[j] = dgdt * (2.0 / span); // chain rule dt/dx = 2/span
        }
        g
    }

    /// Draw `n` independence proposals from the tempered surrogate density
    /// `exp(-(f - f_min)/T)` by per-coordinate inverse-CDF sampling on an
    /// `grid_m`-point grid. Each coordinate is sampled independently (the
    /// separable factorisation), with a uniform within-cell jitter for
    /// continuity. Returns an `n x d` array of points inside the box.
    pub fn sample<R: Rng + ?Sized>(
        &self,
        n: usize,
        temperature: f64,
        grid_m: usize,
        rng: &mut R,
    ) -> Array2<f64> {
        let dim = self.bounds.dims;
        let temp = temperature.max(1e-12);
        let mut out = Array2::<f64>::zeros((n, dim));
        let grid_m = grid_m.max(2);
        for j in 0..dim {
            let (lo, hi) = (self.bounds.low[j], self.bounds.high[j]);
            let span = hi - lo;
            // grid coordinates and energies for this axis
            let mut energy = vec![0.0f64; grid_m];
            let mut emin = f64::INFINITY;
            for (g, e) in energy.iter_mut().enumerate() {
                let frac = g as f64 / (grid_m - 1) as f64;
                let t = 2.0 * frac - 1.0;
                let value = cheb_basis(t, self.degree).dot(&self.coeffs.row(j));
                *e = value;
                if value.is_finite() {
                    emin = emin.min(value);
                }
            }
            if !emin.is_finite() {
                for s in 0..n {
                    out[[s, j]] = lo + rng.random::<f64>() * span;
                }
                continue;
            }
            // tempered weights and their normalised CDF
            let mut cdf = vec![0.0f64; grid_m];
            let mut acc = 0.0;
            for g in 0..grid_m {
                let weight = if energy[g].is_finite() {
                    let z = ((energy[g] - emin) / temp).clamp(0.0, 700.0);
                    (-z).exp()
                } else {
                    0.0
                };
                if weight.is_finite() {
                    acc += weight;
                }
                cdf[g] = acc;
            }
            if !acc.is_finite() || acc <= 0.0 {
                for s in 0..n {
                    out[[s, j]] = lo + rng.random::<f64>() * span;
                }
                continue;
            }
            let cell = if grid_m > 1 {
                span / (grid_m - 1) as f64
            } else {
                span
            };
            for s in 0..n {
                let u = rng.random::<f64>() * acc;
                // first grid index with cdf >= u
                let mut idx = match cdf.binary_search_by(|v| v.total_cmp(&u)) {
                    Ok(i) => i,
                    Err(i) => i,
                };
                if idx >= grid_m {
                    idx = grid_m - 1;
                }
                let frac = idx as f64 / (grid_m - 1) as f64;
                let centre = lo + frac * span;
                let jitter = (rng.random::<f64>() - 0.5) * cell;
                out[[s, j]] = (centre + jitter).clamp(lo, hi);
            }
        }
        out
    }
}

impl Objective<f64> for AdditiveSurrogate {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value(x)
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn dim(&self) -> usize {
        self.bounds.dims
    }
}

impl Gradient<f64> for AdditiveSurrogate {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        // The public `gradient` method is the analytic native gradient
        // (Ceres style).  We delegate for the trait.
        self.gradient(x)
    }
    fn dim(&self) -> usize {
        self.bounds.dims
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn box_bounds(dim: usize, lo: f64, hi: f64) -> Bounds<f64> {
        Bounds::new(Array1::from_elem(dim, lo), Array1::from_elem(dim, hi), 0.0)
    }

    // Styblinski-Tang is separable: 0.5 sum(x^4 - 16 x^2 + 5 x).
    fn styb(x: ArrayView1<f64>) -> f64 {
        0.5 * x
            .iter()
            .map(|&v| v.powi(4) - 16.0 * v * v + 5.0 * v)
            .sum::<f64>()
    }

    #[test]
    fn fit_recovers_separable_value() {
        let dim = 5;
        let bounds = box_bounds(dim, -5.0, 5.0);
        let mut rng = StdRng::seed_from_u64(0);
        let n = 400;
        let mut x = Array2::<f64>::zeros((n, dim));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            for j in 0..dim {
                x[[i, j]] = rng.random::<f64>() * 10.0 - 5.0;
            }
            y[i] = styb(x.row(i));
        }
        let surr = AdditiveSurrogate::fit(x.view(), y.view(), bounds, 8);
        // surrogate value tracks the true separable objective at fresh points
        let mut max_rel = 0.0f64;
        for _ in 0..50 {
            let mut p = Array1::<f64>::zeros(dim);
            for j in 0..dim {
                p[j] = rng.random::<f64>() * 10.0 - 5.0;
            }
            let truth = styb(p.view());
            let approx = surr.value(p.view());
            let rel = (truth - approx).abs() / (1.0 + truth.abs());
            max_rel = max_rel.max(rel);
        }
        assert!(max_rel < 1e-6, "separable fit rel-err too large: {max_rel}");
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let dim = 4;
        let bounds = box_bounds(dim, -5.0, 5.0);
        let mut rng = StdRng::seed_from_u64(1);
        let n = 300;
        let mut x = Array2::<f64>::zeros((n, dim));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            for j in 0..dim {
                x[[i, j]] = rng.random::<f64>() * 10.0 - 5.0;
            }
            y[i] = styb(x.row(i));
        }
        let surr = AdditiveSurrogate::fit(x.view(), y.view(), bounds, 8);
        let p = Array1::from_vec(vec![1.0, -2.0, 0.5, 3.0]);
        let g = surr.gradient(p.view());
        let h = 1e-6;
        for j in 0..dim {
            let mut pp = p.clone();
            let mut pm = p.clone();
            pp[j] += h;
            pm[j] -= h;
            let fd = (surr.value(pp.view()) - surr.value(pm.view())) / (2.0 * h);
            assert!((g[j] - fd).abs() < 1e-3, "grad[{j}] {} vs fd {}", g[j], fd);
        }
    }

    #[test]
    fn tempered_sample_concentrates_at_low_energy() {
        // At a low temperature the separable sampler should place each
        // coordinate near the Styblinski-Tang single-well minimum -2.903534.
        let dim = 6;
        let bounds = box_bounds(dim, -5.0, 5.0);
        let mut rng = StdRng::seed_from_u64(2);
        let n = 500;
        let mut x = Array2::<f64>::zeros((n, dim));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            for j in 0..dim {
                x[[i, j]] = rng.random::<f64>() * 10.0 - 5.0;
            }
            y[i] = styb(x.row(i));
        }
        let surr = AdditiveSurrogate::fit(x.view(), y.view(), bounds, 8);
        let draws = surr.sample(2000, 0.5, 129, &mut rng);
        let mean0 = draws.column(0).mean().unwrap();
        // the global single-well min dominates the tempered density
        assert!(
            (mean0 - (-2.903534)).abs() < 0.7,
            "low-T sample mean {mean0} not near the well -2.9035"
        );
    }

    #[test]
    fn tempered_sample_handles_nonfinite_axis_energy() {
        let dim = 2;
        let bounds = box_bounds(dim, -1.0, 1.0);
        let surr = AdditiveSurrogate {
            bounds,
            intercept: 0.0,
            coeffs: Array2::from_shape_vec((dim, 2), vec![f64::NAN, 0.0, 0.0, 0.0]).unwrap(),
            col_means: Array2::zeros((dim, 2)),
            degree: 2,
        };
        let mut rng = StdRng::seed_from_u64(3);

        let draws = surr.sample(32, 0.5, 17, &mut rng);

        assert_eq!(draws.nrows(), 32);
        assert_eq!(draws.ncols(), dim);
        assert!(draws.iter().all(|v| v.is_finite()));
        assert!(draws.iter().all(|v| (-1.0..=1.0).contains(v)));
    }
}
