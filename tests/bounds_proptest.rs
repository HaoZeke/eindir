use eindir_core::Bounds;
use ndarray::Array1;
use proptest::prelude::*;
use rand::SeedableRng;

#[test]
#[should_panic(expected = "low must not exceed high")]
fn reversed_bounds_are_rejected() {
    Bounds::new(
        Array1::from_vec(vec![1.0_f64]),
        Array1::from_vec(vec![-1.0]),
        0.0,
    );
}

#[test]
#[should_panic(expected = "slack must be non-negative")]
fn negative_slack_is_rejected() {
    Bounds::new(
        Array1::from_vec(vec![-1.0_f64]),
        Array1::from_vec(vec![1.0]),
        -1e-9,
    );
}

#[test]
fn contains_rejects_wrong_dimensions_without_panicking() {
    let b = Bounds::new(
        Array1::from_vec(vec![-1.0_f64, -1.0]),
        Array1::from_vec(vec![1.0, 1.0]),
        0.0,
    );
    assert!(!b.contains(Array1::from_vec(vec![0.0]).view()));
    assert!(!b.contains(Array1::from_vec(vec![0.0, 0.0, 0.0]).view()));
}

proptest! {
    #[test]
    fn mkpoint_always_in_bounds(seed in any::<u64>()) {
        let b = Bounds::new(
            Array1::from_vec(vec![-1.0_f64, -1.0]),
            Array1::from_vec(vec![1.0,       1.0]),
            1e-9,
        );
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        for _ in 0..100 {
            let p = b.mkpoint(&mut rng);
            prop_assert!(b.contains(p.view()));
        }
    }

    #[test]
    fn clip_always_in_bounds(
        x in proptest::collection::vec(-10.0_f64..10.0, 2..=2)
    ) {
        let b = Bounds::new(
            Array1::from_vec(vec![-1.0, -1.0]),
            Array1::from_vec(vec![ 1.0,  1.0]),
            0.0,
        );
        let arr = Array1::from_vec(x);
        prop_assert!(b.contains(b.clip(arr.view()).view()));
    }
}
