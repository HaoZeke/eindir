``low_discrepancy_points`` and ``shifted_low_discrepancy_points`` raise ``ValueError`` when a bound is not a number, from ``Bounds::try_new``, instead of ``PanicException``.
Shape and ordering faults use that error's text, and a NaN coordinate is reported before an inverted pair.
