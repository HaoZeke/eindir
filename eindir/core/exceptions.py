class OutOfBounds(ValueError):
    """
    Exception raised when a point is out of bounds.

    #### Description
    This class is a subclass of the built-in `ValueError` exception in Python.
    It is raised when a point is outside the defined bounds.

    #### Parameters
    **message** (`str`)
    : The error message to be displayed when the exception is raised.

    #### Attributes
    **message** (`str`)
    : The error message associated with the exception.
    """

    def __init__(self, message):
        """
        Initializes an instance of the OutOfBounds class.

        #### Parameters
        **message** (`str`)
        : The error message to be displayed when the exception is raised.
        """
        self.message = message
        super().__init__(self.message)
