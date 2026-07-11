use eindir_core::{
    Bounds, boundary_anchored_low_discrepancy_points, low_discrepancy_points, radical_inverse,
    shifted_low_discrepancy_points,
};
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

#[test]
fn shifted_low_discrepancy_points_are_replicated_and_bounded() {
    let bounds = Bounds::new(
        Array1::from_vec(vec![-1.0, -2.0]),
        Array1::from_vec(vec![1.0, 2.0]),
        0.0,
    );

    let shifted = shifted_low_discrepancy_points(&bounds, 16, 1, 7);
    let repeated = shifted_low_discrepancy_points(&bounds, 16, 1, 7);
    let base = low_discrepancy_points(&bounds, 16, 1);

    assert_eq!(shifted, repeated);
    assert_ne!(shifted, base);
    assert_eq!(shifted.shape(), &[16, 2]);
    for row in shifted.outer_iter() {
        assert!(bounds.contains(row));
    }
}

#[test]
fn boundary_anchored_points_start_with_vertices_when_the_design_fits() {
    let bounds = Bounds::new(
        Array1::from_vec(vec![-1.0, -2.0]),
        Array1::from_vec(vec![1.0, 2.0]),
        0.0,
    );

    let points = boundary_anchored_low_discrepancy_points(&bounds, 5, 1);

    assert_eq!(points.row(0).to_vec(), vec![-1.0, -2.0]);
    assert_eq!(points.row(1).to_vec(), vec![1.0, -2.0]);
    assert_eq!(points.row(2).to_vec(), vec![-1.0, 2.0]);
    assert_eq!(points.row(3).to_vec(), vec![1.0, 2.0]);
    assert_eq!(points.row(4).to_vec(), vec![0.0, 0.0]);
}

#[test]
fn boundary_anchored_points_use_diagonal_vertices_for_large_vertex_sets() {
    let bounds = Bounds::new(
        Array1::from_vec(vec![-1.0, -2.0, -3.0]),
        Array1::from_vec(vec![1.0, 2.0, 3.0]),
        0.0,
    );

    let points = boundary_anchored_low_discrepancy_points(&bounds, 4, 1);

    assert_eq!(points.row(0).to_vec(), vec![-1.0, -2.0, -3.0]);
    assert_eq!(points.row(1).to_vec(), vec![1.0, 2.0, 3.0]);
    assert_eq!(points.row(2).to_vec(), vec![0.0, 0.0, 0.0]);
    for row in points.outer_iter() {
        assert!(bounds.contains(row));
    }
}
