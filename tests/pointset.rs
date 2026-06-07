use eindir_core::{low_discrepancy_points, radical_inverse, Bounds};
use ndarray::Array1;

#[test]
fn radical_inverse_matches_base_two_van_der_corput_prefix() {
    let prefix: Vec<f64> = (1..=8).map(|idx| radical_inverse(idx, 2)).collect();

    assert_eq!(
        prefix,
        vec![0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625]
    );
}

#[test]
fn low_discrepancy_points_are_deterministic_and_bounded() {
    let bounds = Bounds::new(
        Array1::from_vec(vec![-1.0, -2.0]),
        Array1::from_vec(vec![1.0, 2.0]),
        0.0,
    );

    let first = low_discrepancy_points(&bounds, 16, 1);
    let second = low_discrepancy_points(&bounds, 16, 1);

    assert_eq!(first, second);
    assert_eq!(first.shape(), &[16, 2]);
    for row in first.outer_iter() {
        assert!(bounds.contains(row));
    }
}
