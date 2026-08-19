Drop the blanket ``DifferentiableObjective`` impl so a type can override
``value_and_gradient`` for a fused ``(f, grad)`` evaluation. Builtins now
carry an explicit empty impl; downstream crates must do the same.
