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

#[test]
#[should_panic(expected = "point has the wrong dimension")]
fn clip_rejects_wrong_dimensions() {
    let bounds = Bounds::new(
        Array1::from_vec(vec![0.0, 0.0]),
        Array1::from_vec(vec![1.0, 1.0]),
        0.0,
    );
    let _ = bounds.clip(Array1::from_vec(vec![0.5]).view());
}

mod try_new {
    use eindir_core::{Bounds, BoundsError};
    use ndarray::Array1;

    fn attempt(low: &[f64], high: &[f64], slack: f64) -> Result<Bounds<f64>, BoundsError> {
        Bounds::try_new(
            Array1::from_vec(low.to_vec()),
            Array1::from_vec(high.to_vec()),
            slack,
        )
    }

    #[test]
    fn a_valid_box_is_built() {
        let b = attempt(&[-1.0, 0.0], &[1.0, 0.0], 1e-9).expect("valid");
        assert_eq!(b.dims, 2);
        assert_eq!(b.slack, 1e-9);
    }

    #[test]
    fn sides_of_different_lengths_are_named() {
        assert_eq!(
            attempt(&[0.0, 0.0], &[1.0], 0.0).unwrap_err(),
            BoundsError::ShapeMismatch { low: 2, high: 1 }
        );
    }

    #[test]
    fn the_first_inverted_coordinate_is_named() {
        assert_eq!(
            attempt(&[0.0, 2.0, 5.0], &[1.0, 1.0, 4.0], 0.0).unwrap_err(),
            BoundsError::Inverted { index: 1 }
        );
    }

    #[test]
    fn a_slack_that_is_negative_or_not_a_number_is_refused() {
        for slack in [-1e-9, f64::NAN, f64::NEG_INFINITY] {
            assert_eq!(
                attempt(&[0.0], &[1.0], slack).unwrap_err(),
                BoundsError::NegativeSlack
            );
        }
    }

    #[test]
    fn a_coordinate_that_is_not_a_number_is_refused() {
        assert_eq!(
            attempt(&[0.0, f64::NAN], &[1.0, 1.0], 0.0).unwrap_err(),
            BoundsError::NotANumber { index: 1 }
        );
        assert_eq!(
            attempt(&[0.0], &[f64::NAN], 0.0).unwrap_err(),
            BoundsError::NotANumber { index: 0 }
        );
    }

    #[test]
    fn new_panics_with_the_message_try_new_returns() {
        let cases: [(&[f64], &[f64], f64); 4] = [
            (&[0.0, 0.0], &[1.0], 0.0),
            (&[1.0], &[0.0], 0.0),
            (&[0.0], &[1.0], -1.0),
            (&[f64::NAN], &[1.0], 0.0),
        ];
        for (low, high, slack) in cases {
            let message = attempt(low, high, slack).unwrap_err().to_string();
            let caught = std::panic::catch_unwind(|| {
                Bounds::new(
                    Array1::from_vec(low.to_vec()),
                    Array1::from_vec(high.to_vec()),
                    slack,
                )
            })
            .expect_err("new panics");
            let text = caught
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| caught.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                .expect("a message");
            assert_eq!(text, message);
        }
    }

    #[test]
    fn single_precision_boxes_use_the_same_checks() {
        let low = Array1::from_vec(vec![1.0_f32]);
        let high = Array1::from_vec(vec![0.0_f32]);
        assert_eq!(
            Bounds::try_new(low, high, 0.0).unwrap_err(),
            BoundsError::Inverted { index: 0 }
        );
    }
}
