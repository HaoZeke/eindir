``Bounds::try_new`` builds a box and returns a ``non_exhaustive`` ``BoundsError`` for sides of different lengths, an inverted pair, a coordinate that is not a number or a slack that is negative or not a number, where ``Bounds::new`` panics.
``Bounds::new`` now calls it and panics with the message of the error it returns.
The Python ``Bounds`` constructor raises ``ValueError`` with that message instead of a ``PanicException``.
