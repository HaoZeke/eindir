//! Generalized Langevin equation (GLE) colored-noise thermostat, in the
//! Markovian extended-phase-space form of Ceriotti, Bussi and Parrinello.
//!
//! White-noise Langevin dynamics has a single friction `gamma`, so it can
//! critically damp only one frequency; across a spectrum of curvatures (an
//! ill-conditioned objective) most modes are far from critical and decorrelate
//! slowly. The GLE replaces the white noise by a colored noise generated from
//! `ns` auxiliary momenta, whose drift matrix `A` shapes a frequency-dependent
//! friction. A well-chosen `A` makes the sampling efficiency nearly flat across
//! a target frequency band -- "optimal sampling" -- so conditioning is handled
//! by the noise spectrum exactly as the `1/sqrt(D)` scale handles dimension.
//!
//! The per-step propagator follows i-PI's `ThermoGLE`:
//!   `T = exp(-dt A)`,  `S S^T = k_B (C - T C T^T)`,  `s <- T s + S xi`,
//! with `xi` standard normal and `s[0]` the physical (mass-scaled) momentum.
//! For canonical sampling `C = temperature * I`, which makes the auxiliary
//! process leave the Boltzmann momentum distribution invariant for ANY valid
//! drift `A` -- the "a la carte" decoupling of correctness (set by `C`) from
//! acceleration (set by `A`). `matrix_exp` is a Taylor scaling-and-squaring and
//! the noise factor is an LDL^T square root, so no linear-algebra backend is
//! needed.

use ndarray::{Array1, Array2, ArrayViewMut2};
use rand::Rng;
use rand_distr::StandardNormal;

/// Matrix exponential by Taylor series with scaling and squaring.
///
/// `exp(M) = (exp(M / 2^k))^{2^k}` with the inner exponential a truncated Taylor
/// series evaluated by Horner. Matches i-PI's `matrix_exp`; for the small GLE
/// drift matrices (dimension `ns + 1`, a handful of rows) it is exact to machine
/// precision with `n_taylor = 20`, `n_square = 10`.
pub fn matrix_exp(m: &Array2<f64>, n_taylor: usize, n_square: usize) -> Array2<f64> {
    let n = m.nrows();
    assert_eq!(n, m.ncols(), "matrix_exp needs a square matrix");
    // Taylor coefficients 1, 1, 1/2, 1/6, ...
    let mut tc = vec![0.0f64; n_taylor + 1];
    tc[0] = 1.0;
    for i in 0..n_taylor {
        tc[i + 1] = tc[i] / (i as f64 + 1.0);
    }
    let scale = 2f64.powi(n_square as i32);
    let sm = m / scale;
    let mut em = Array2::<f64>::eye(n) * tc[n_taylor];
    for i in (0..n_taylor).rev() {
        em = sm.dot(&em);
        em = em + Array2::<f64>::eye(n) * tc[i];
    }
    for _ in 0..n_square {
        em = em.dot(&em);
    }
    em
}

/// Stabilised LDL^T square root: returns lower-triangular `S` with `S S^T = M`
/// for a symmetric positive-semidefinite `M`, zeroing any negative pivot. This
/// is i-PI's `stab_cholesky`; it needs no eigensolver and is numerically robust
/// for the GLE noise covariance, which can be singular at small `dt`.
pub fn ldl_sqrt(m: &Array2<f64>) -> Array2<f64> {
    let n = m.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    let mut d = vec![0.0f64; n];
    for i in 0..n {
        l[[i, i]] = 1.0;
        for j in 0..i {
            let mut v = m[[i, j]];
            for k in 0..j {
                v -= l[[i, k]] * l[[j, k]] * d[k];
            }
            if d[j] != 0.0 {
                v /= d[j];
            }
            l[[i, j]] = v;
        }
        let mut dii = m[[i, i]];
        for k in 0..i {
            dii -= l[[i, k]] * l[[i, k]] * d[k];
        }
        d[i] = dii;
    }
    // S = L D^{1/2}: column j is scaled by sqrt(d[j]) (zeroing negative pivots).
    let sqrt_d: Vec<f64> = d.iter().map(|&v| if v > 0.0 { v.sqrt() } else { 0.0 }).collect();
    let mut s = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            s[[i, j]] += l[[i, j]] * sqrt_d[j];
        }
    }
    s
}

/// A GLE thermostat with precomputed drift and noise factors for one timestep.
pub struct GleThermostat {
    /// Number of auxiliary momenta (matrix dimension is `ns + 1`).
    pub ns: usize,
    /// Deterministic propagator `T = exp(-dt A)`.
    t: Array2<f64>,
    /// Noise factor `S` with `S S^T = k_B (C - T C T^T)`.
    s: Array2<f64>,
}

impl GleThermostat {
    /// Build from an explicit drift `a` and covariance `c` at timestep `dt`
    /// (Boltzmann constant `kb`, in reduced units `kb = 1`). `a` and `c` are
    /// `(ns+1) x (ns+1)`.
    pub fn new(a: &Array2<f64>, c: &Array2<f64>, dt: f64, kb: f64) -> Self {
        let n = a.nrows();
        assert_eq!(n, a.ncols(), "drift matrix must be square");
        assert_eq!(c.shape(), a.shape(), "covariance must match drift shape");
        let t = matrix_exp(&(a * (-dt)), 20, 10);
        let sst = (c - &t.dot(c).dot(&t.t())) * kb;
        // symmetrise to kill round-off asymmetry before the LDL^T root
        let sst = (&sst + &sst.t()) * 0.5;
        let s = ldl_sqrt(&sst);
        Self { ns: n - 1, t, s }
    }

    /// Canonical thermostat: `C = temperature * I`, leaving the Boltzmann
    /// momentum law invariant for any valid drift `a`.
    pub fn canonical(a: &Array2<f64>, dt: f64, temperature: f64, kb: f64) -> Self {
        let n = a.nrows();
        let c = Array2::<f64>::eye(n) * temperature;
        Self::new(a, &c, dt, kb)
    }

    /// Advance the auxiliary state one timestep in place. `s` is `(ns+1) x dim`;
    /// row 0 holds the physical (mass-scaled) momentum of each coordinate, rows
    /// `1..=ns` the auxiliary momenta. Applies `s <- T s + S xi`.
    pub fn step<R: Rng + ?Sized>(&self, s: &mut ArrayViewMut2<f64>, rng: &mut R) {
        let dim = s.ncols();
        let n = self.ns + 1;
        let mut xi = Array2::<f64>::zeros((n, dim));
        for v in xi.iter_mut() {
            *v = rng.sample(StandardNormal);
        }
        let next = self.t.dot(s) + self.s.dot(&xi);
        s.assign(&next);
    }

    /// Draw an auxiliary state from its stationary covariance `C` (mass-scaled
    /// physical momentum in row 0), used to initialise a chain at equilibrium.
    pub fn sample_stationary<R: Rng + ?Sized>(
        &self,
        c: &Array2<f64>,
        dim: usize,
        kb: f64,
        rng: &mut R,
    ) -> Array2<f64> {
        let n = self.ns + 1;
        let sc = ldl_sqrt(&(c * kb));
        let mut xi = Array2::<f64>::zeros((n, dim));
        for v in xi.iter_mut() {
            *v = rng.sample(StandardNormal);
        }
        sc.dot(&xi)
    }
}

/// Construct an optimal-sampling drift matrix over a target frequency band.
///
/// Each of `ns` auxiliary degrees of freedom is a damped oscillator placed at a
/// log-spaced frequency in `[omega_min, omega_max]`, coupled to the physical
/// momentum by a skew block. The drift is `A = D + W` with `D` the diagonal
/// dissipation (a small white friction on the physical momentum plus the
/// per-bath rates `omega_k`) and `W` skew, so `A + A^T = 2 D` is positive
/// semidefinite and the fluctuation-dissipation relation `A C + C A^T = B B^T`
/// holds for `C = I` (canonical sampling) with a valid `B B^T = 2 D`. The
/// Lorentzian friction each bath contributes peaks at its `omega_k`; spreading
/// the baths log-uniformly flattens the friction -- hence the sampling
/// efficiency -- across the band, the colored-noise analogue of critical
/// damping over a whole spectrum rather than at one frequency.
pub fn optimal_sampling_drift(omega_min: f64, omega_max: f64, ns: usize) -> Array2<f64> {
    assert!(ns >= 1, "need at least one auxiliary DOF");
    assert!(
        omega_max > omega_min && omega_min > 0.0,
        "require 0 < omega_min < omega_max"
    );
    let n = ns + 1;
    let mut a = Array2::<f64>::zeros((n, n));
    // log-spaced bath frequencies
    let log_lo = omega_min.ln();
    let log_hi = omega_max.ln();
    // a small white friction anchors the physical-momentum dissipation
    a[[0, 0]] = omega_min;
    for k in 0..ns {
        let frac = if ns == 1 {
            0.5
        } else {
            k as f64 / (ns as f64 - 1.0)
        };
        let omega_k = (log_lo + frac * (log_hi - log_lo)).exp();
        // coupling chosen so each bath contributes comparable friction area
        let c_k = omega_k.sqrt() * (omega_max / omega_min).powf(1.0 / (2.0 * ns as f64));
        let idx = k + 1;
        a[[idx, idx]] = omega_k; // bath dissipation rate
        a[[0, idx]] = c_k; // skew coupling: physical <- aux
        a[[idx, 0]] = -c_k; // skew coupling: aux <- physical
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn matrix_exp_diagonal_and_zero() {
        let z = Array2::<f64>::zeros((3, 3));
        let e0 = matrix_exp(&z, 20, 10);
        assert!((&e0 - &Array2::<f64>::eye(3)).iter().all(|v| v.abs() < 1e-12));
        let d = Array2::from_diag(&Array1::from_vec(vec![0.5, -1.0, 2.0]));
        let ed = matrix_exp(&d, 20, 10);
        for (i, val) in [0.5f64.exp(), (-1.0f64).exp(), 2.0f64.exp()].iter().enumerate() {
            assert!((ed[[i, i]] - val).abs() < 1e-10, "exp diag {i}");
        }
    }

    #[test]
    fn ldl_sqrt_reconstructs() {
        // a symmetric positive-definite matrix
        let m = ndarray::array![[4.0, 2.0, 0.5], [2.0, 3.0, 1.0], [0.5, 1.0, 2.0]];
        let s = ldl_sqrt(&m);
        let rebuilt = s.dot(&s.t());
        assert!((&rebuilt - &m).iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn optimal_drift_satisfies_fluctuation_dissipation() {
        // A + A^T must be PSD so that B B^T = A C + C A^T (C = I) is a valid
        // diffusion; the construction makes A + A^T diagonal and non-negative.
        let a = optimal_sampling_drift(0.1, 10.0, 4);
        let sym = &a + &a.t();
        for i in 0..sym.nrows() {
            assert!(sym[[i, i]] >= -1e-12, "diagonal {i} negative");
            for j in 0..sym.ncols() {
                if i != j {
                    assert!(sym[[i, j]].abs() < 1e-12, "off-diagonal not skew at {i},{j}");
                }
            }
        }
    }

    #[test]
    fn canonical_thermostat_preserves_covariance() {
        // Starting from the stationary covariance C = I, one GLE step must leave
        // the covariance invariant: E[s' s'^T] = T C T^T + S S^T = C. Estimate
        // the covariance over many independent draws and check row 0 (the
        // physical momentum) keeps unit variance.
        let a = optimal_sampling_drift(0.2, 5.0, 3);
        let dt = 0.05;
        let temp = 1.0;
        let gle = GleThermostat::canonical(&a, dt, temp, 1.0);
        let c = Array2::<f64>::eye(a.nrows()) * temp;
        let mut rng = StdRng::seed_from_u64(0);
        let trials = 40000;
        let mut var0 = 0.0;
        for _ in 0..trials {
            let mut s = gle.sample_stationary(&c, 1, 1.0, &mut rng);
            gle.step(&mut s.view_mut(), &mut rng);
            var0 += s[[0, 0]] * s[[0, 0]];
        }
        var0 /= trials as f64;
        // physical-momentum variance stays at temperature (canonical)
        assert!((var0 - temp).abs() < 0.05, "momentum variance drifted: {var0}");
    }
}
