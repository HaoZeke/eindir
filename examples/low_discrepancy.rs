//! Halton designs in a box: the plain design, a seeded shifted replica and a
//! design anchored at the corners. The low-discrepancy tutorial includes
//! this file; `cargo run --example low_discrepancy`.
use eindir_core::{
    Bounds, boundary_anchored_low_discrepancy_points, halton_unit, low_discrepancy_points,
    shifted_low_discrepancy_points,
};
use ndarray::array;

fn main() {
    // Axis i of Halton point k is the radical inverse of k in the i-th prime.
    assert_eq!(halton_unit(1, 2), array![0.5, 1.0 / 3.0]);

    let bounds = Bounds::new(array![-5.12, -5.12], array![5.12, 5.12], 0.0);

    let starts = low_discrepancy_points(&bounds, 16, 1);
    assert_eq!(starts.shape(), &[16, 2]);
    assert!(starts.outer_iter().all(|x| bounds.contains(x)));
    println!("first start {}", starts.row(0));

    // The same seed gives the same replica; a different seed shifts the design.
    let a = shifted_low_discrepancy_points(&bounds, 16, 1, 3);
    let b = shifted_low_discrepancy_points(&bounds, 16, 1, 3);
    let c = shifted_low_discrepancy_points(&bounds, 16, 1, 4);
    assert_eq!(a, b);
    assert_ne!(a, c);

    // Four corners and the centre fit in the first five rows of a 2-D design.
    let anchored = boundary_anchored_low_discrepancy_points(&bounds, 16, 1);
    assert_eq!(anchored.row(0), array![-5.12, -5.12]);
    assert_eq!(anchored.row(3), array![5.12, 5.12]);
    assert_eq!(anchored.row(4), array![0.0, 0.0]);
    assert_eq!(anchored.row(5), starts.row(5));
    println!("anchored rows 0..5 hold the corners and the centre");
}
