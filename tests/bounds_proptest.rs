use eindir_core::Bounds;
use ndarray::Array1;
use proptest::prelude::*;
use rand::SeedableRng;

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
