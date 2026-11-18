``PanicOnError::value_and_gradient`` now panics with ``"value and gradient evaluation failed"`` instead of ``"objective evaluation failed"``, distinguishing gradient and fused failures from ``PanicOnError::eval``.
Fallible C callback messages are now decoded losslessly into valid UTF-8 strings across ``CallbackError``, ``eindir_last_error()``, and ``message_out`` by doubling backslashes and escaping invalid UTF-8 bytes as lowercase ``\xhh`` sequences.
Fixed four ``rustdoc`` link warnings on ``eindir_objective_t``, ``eindir_objective_bounds_low``, and ``eindir_objective_bounds_high``.
