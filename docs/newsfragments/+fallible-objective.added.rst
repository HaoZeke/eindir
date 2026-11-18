Fallible objectives and gradients, additively.
``TryObjective``, ``TryGradient`` and ``TryDifferentiableObjective`` have ``try_eval``, ``try_grad`` and ``try_value_and_gradient`` and an associated error type that a driver returns unchanged.
Every ``Objective``, ``Gradient`` and ``DifferentiableObjective`` is also the matching ``Try*`` trait through a blanket impl, with ``Error = core::convert::Infallible``.
``NanOnError`` and ``PanicOnError`` adapt back to the infallible traits, and ``TryOracle``, ``TryFiniteDiffGradient`` and ``try_eval_batch_parallel`` build and evaluate fallible objectives.
The ``capi`` feature adds ``eindir_try_objective_t`` with callbacks that return an ``int32_t`` status and write a message, ``eindir_try_objective_eval`` and ``_grad``, and the Rust ``EindirTryObjectiveWrapper`` with ``CallbackError``.
``eindir_objective_t``, its callbacks and its layout are unchanged.
``eindir_core_abi_stamp()`` reports ABI minor 2 and the feature bit ``EINDIR_ABI_FEATURE_TRY_CALLBACKS``.
The design note is the Fallible objectives page of the manual.
