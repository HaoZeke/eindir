``Bounds::new`` panics when a coordinate of ``low`` or ``high`` is not a number.
Only that rejection is new.
On main, ``assert!(slack >= T::zero())`` already panics for a NaN slack, because the comparison is false.
A box with a NaN side contains no point.
