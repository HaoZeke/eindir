The ``autodiff`` feature re-exports ``num_dual::Dual64`` and ``DualNum`` and adds ``directional_derivative``.
A map ``R^n -> R^m`` written on that scalar dual number returns its value and the Jacobian-vector product along one seed.
An empty point, a non-finite input, or a non-finite output is ``Error::OutOfBounds``.
``ForwardAd`` is unchanged and still differentiates a scalar field with one ``DualDVec64`` sweep.
