//! An objective whose evaluation can fail: the error reaches a generic
//! driver unchanged, and an infallible objective passes through the same
//! driver. The fallible-objective how-to includes this file;
//! `cargo run --example fallible_objective`.
use eindir_core::{
    Bounds, NanOnError, Objective, Rastrigin, TryDifferentiableObjective, TryOracle,
};
use ndarray::{Array1, ArrayView1, array};

#[derive(Debug, PartialEq)]
enum EngineError {
    Refused { x0: f64 },
}

/// One steepest-descent step that returns the objective's own error.
fn descend<O>(obj: &O, x: &mut Array1<f64>, step: f64) -> Result<f64, O::Error>
where
    O: TryDifferentiableObjective<f64>,
{
    let (_, g) = obj.try_value_and_gradient(x.view())?;
    *x = &*x - &(g * step);
    let (value, _) = obj.try_value_and_gradient(x.view())?;
    Ok(value)
}

fn main() {
    let bounds = Bounds::new(array![-5.0, -5.0], array![5.0, 5.0], 0.0);
    let engine = TryOracle::new(bounds, |x: ArrayView1<f64>| {
        if x[0] > 4.0 {
            return Err(EngineError::Refused { x0: x[0] });
        }
        Ok((x.dot(&x), &x * 2.0))
    });

    let mut x = array![1.0, 1.0];
    assert_eq!(descend(&engine, &mut x, 0.25), Ok(0.5));

    let mut far = array![4.5, 0.0];
    assert_eq!(
        descend(&engine, &mut far, 0.25),
        Err(EngineError::Refused { x0: 4.5 })
    );

    // An infallible objective is a fallible one with Error = Infallible.
    let mut y = array![0.1, 0.0];
    let value = descend(&Rastrigin::<2>::new(), &mut y, 0.001).unwrap_or_else(|e| match e {});
    println!("Rastrigin after one step: {value:.4}");

    // Back to the infallible traits, keeping each error in a log.
    let log = std::sync::Mutex::new(Vec::new());
    let plain = NanOnError::new(engine, |e| log.lock().unwrap().push(e));
    assert!(plain.eval(array![4.5, 0.0].view()).is_nan());
    println!("logged: {:?}", log.lock().unwrap());
}
