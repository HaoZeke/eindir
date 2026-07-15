//! Wall-clock of eval_batch_parallel vs thread count (run with RAYON_NUM_THREADS).
use eindir_core::{Objective, Rastrigin, eval_batch_parallel, low_discrepancy_points};
use std::time::Instant;

fn main() {
    let obj = Rastrigin::<16>::new();
    let x = low_discrepancy_points(obj.bounds(), 4096, 1);
    // warmup
    let _ = eval_batch_parallel(&obj, x.view());
    let t0 = Instant::now();
    let iters = 50usize;
    for _ in 0..iters {
        let _ = eval_batch_parallel(&obj, x.view());
    }
    let dt = t0.elapsed().as_secs_f64();
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "auto".into());
    println!(
        "threads={threads} n=4096 dim=16 iters={iters} total_s={dt:.6} per_batch_ms={:.3}",
        1e3 * dt / iters as f64
    );
}
