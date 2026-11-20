``Bounds::new`` panics on a coordinate of ``low`` or ``high`` that is not a number, and on a NaN slack.
It accepted both before, and a box with a NaN side contains no point.
