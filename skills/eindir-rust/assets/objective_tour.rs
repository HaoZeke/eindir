//! An objective with box bounds, Halton start points, batch evaluation,
//! gradients and a builtin test function, with eindir-core.

use std::sync::atomic::{AtomicUsize, Ordering};

use eindir_core::{
    Bounds, FiniteDiffGradient, Gradient, Objective, Rastrigin, eval_batch_parallel,
    low_discrepancy_points,
};
use ndarray::{Array1, ArrayView1, array};

/// f(x) = sum_i (x_i - 1)^2 on [-5, 5]^dim, counting its evaluations.
struct ShiftedSphere {
    bounds: Bounds<f64>,
    evals: AtomicUsize,
}

impl ShiftedSphere {
    fn new(dim: usize) -> Self {
        let low = Array1::from_elem(dim, -5.0);
        let high = Array1::from_elem(dim, 5.0);
        Self {
            bounds: Bounds::new(low, high, 0.0),
            evals: AtomicUsize::new(0),
        }
    }
}

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

fn main() {
    let obj = ShiftedSphere::new(3);

    let starts = low_discrepancy_points(obj.bounds(), 16, 1);
    assert_eq!(starts.dim(), (16, 3));
    assert!(starts.outer_iter().all(|row| obj.bounds().contains(row)));

    let serial = obj.eval_batch(starts.view());
    let parallel = eval_batch_parallel(&obj, starts.view());
    assert_eq!(serial, parallel);
    let (best, value) = serial
        .iter()
        .copied()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .expect("at least one start");
    println!(
        "best of 16 starts: {} with f = {value:.4}",
        starts.row(best)
    );

    let fd = FiniteDiffGradient::new(obj);
    let before = fd.obj.evals.load(Ordering::Relaxed);
    let x = array![0.0, 2.0, 3.0];
    let g = fd.grad(x.view());
    let used = fd.obj.evals.load(Ordering::Relaxed) - before;
    assert_eq!(used, 2 * x.len());
    println!(
        "central differences, h = {}: {g:.6} from {used} evaluations",
        fd.h
    );

    let rastrigin = Rastrigin::<2>::new();
    let min = rastrigin
        .global_min()
        .expect("Rastrigin has a known minimum");
    assert_eq!(rastrigin.eval(min.pos.view()), min.val);
    assert_eq!(Objective::dim(&rastrigin), Gradient::dim(&rastrigin));
    println!(
        "Rastrigin<2>: f = {} at {}, gradient {}",
        min.val,
        min.pos,
        rastrigin.grad(min.pos.view())
    );
}
