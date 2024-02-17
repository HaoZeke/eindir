import abc
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from eindir.core.exceptions import OutOfBounds


@dataclass
class FPair:
    """
    Class for handling position-value pairs.

    #### Description
    This class is used to represent a position-value pair where the position is
    an array and the value is a float. It provides a method to evaluate a
    function at the given position and store the result as the value.

    #### Parameters
    **pos** (`npt.NDArray`)
    : An array representing the position.

    **val** (`float`)
    : A float representing the value at the position.

    #### Attributes
    **pos** (`npt.NDArray`)
    : An array representing the position.

    **val** (`float`)
    : A float representing the value at the position. This value is updated
    when the `EvalFunc` method is called.

    #### Notes
    The `EvalFunc` method takes a callable object (a function) as an argument,
    evaluates this function at the position stored in the `pos` attribute, and
    updates the `val` attribute with the result.
    """

    pos: npt.NDArray
    val: float

    def EvalFunc(self, ObjFunc: callable):
        """
        Evaluates a function at the position and updates the value.

        #### Parameters
        **ObjFunc** (`callable`)
        : A function to evaluate at the position. This function should take an
        array (the position) as an argument and return a float (the value).

        #### Notes
        This method evaluates the provided function at the position stored in
        the `pos` attribute and updates the `val` attribute with the result.
        """
        self.val = ObjFunc(self.pos)


@dataclass
class NumLimit:
    """
    Class for tracking function bounds.

    #### Description
    This class is used to represent and enforce numerical limits or bounds on a
    function. It provides methods to check if a given position is within the
    bounds, generate a random point within the bounds, and clip a point to the
    bounds.

    #### Parameters
    **low** (`npt.NDArray`)
    : An array representing the lower bounds that the function must not be
    below.

    **high** (`npt.NDArray`)
    : An array representing the upper bounds that the function must not exceed.

    **slack** (`float`, optional)
    : The amount by which bounds may be exceeded without an error. Default is
    `1e-6`.

    **dims** (`int`, optional)
    : The number of dimensions, which should be the same as the length of the
    `low` and `high` arrays. Default is `1`.

    #### Attributes
    **low** (`npt.NDArray`)
    : An array representing the lower bounds.

    **high** (`npt.NDArray`)
    : An array representing the upper bounds.

    **slack** (`float`)
    : The amount by which bounds may be exceeded without an error.

    **dims** (`int`)
    : The number of dimensions.

    #### Notes
    The `check` method checks if a given position is within the bounds. If the
    position is not within the slack of the bounds, an `OutOfBounds` exception
    is raised.

    The `mkpoint` method generates a random point within the bounds.

    The `clip` method clips a given point to the bounds.
    """

    low: npt.NDArray
    high: npt.NDArray
    slack: float = 1e-6
    dims: int = 1

    def check(self, pos: npt.NDArray):
        """
        Checks if a position is within the bounds.

        #### Parameters
        **pos** (`npt.NDArray`)
        : The position to check.

        #### Notes
        This method checks if the provided position is within the slack of the
        bounds. If the position is not within the slack of the bounds, an
        `OutOfBounds` exception is raised.
        """
        if not (
            np.all(pos > self.low - self.slack) and np.all(pos < self.high + self.slack)
        ):
            raise OutOfBounds(
                f"{pos} is not within {self.slack} of {self.low} and {self.high}"
            )
        return

    def mkpoint(self) -> npt.NDArray:
        """
        Generates a random point within the bounds.

        #### Notes
        This method generates a random point within the bounds using a uniform
        distribution. The number of dimensions of the point is equal to the
        `dims` attribute.

        TODO: Handle other constraints (undefined regions)
        """
        return np.random.default_rng().uniform(low=self.low, high=self.high)

    def clip(self, point: npt.NDArray) -> npt.NDArray:
        """
        Clips a point to the bounds.

        #### Parameters
        **point** (`npt.NDArray`)
        : The point to clip.

        #### Notes
        This method clips the provided point to the bounds. If the point is
        below the lower bound, it is set to the lower bound. If the point is
        above the upper bound, it is set to the upper bound.
        """
        return np.clip(point, self.low, self.high)


class ObjectiveFunction(metaclass=abc.ABCMeta):
    """
    Abstract base class for an objective function.

    #### Description
    This class represents an objective function that can be evaluated at a
    single point or multiple points. It keeps track of the number of function
    calls and enforces numerical limits on the function.

    #### Parameters
    **limits** (`NumLimit`)
    : An instance of the `NumLimit` class representing the numerical limits of
    the function.

    **global_min** (`FPair`, optional)
    : An instance of the `FPair` class representing the global minimum of the
    function. Default is `None`.

    #### Attributes
    **calls** (`int`)
    : The number of times the function has been called.

    **limits** (`NumLimit`)
    : The numerical limits of the function.

    **globmin** (`FPair`)
    : The global minimum of the function.

    #### Notes
    Subclasses must implement the `__call__`, `singlepoint`, `multipoint`, and
    `__repr__` methods. The `__call__` method determines whether to call the
    `singlepoint` or `multipoint` method based on the shape of the input.
    """

    def __init__(self, limits: NumLimit, global_min: FPair = None):
        """
        Initializes an instance of the ObjectiveFunction class.

        #### Parameters
        **limits** (`NumLimit`)
        : An instance of the `NumLimit` class representing the numerical limits
        of the function.

        **global_min** (`FPair`, optional)
        : An instance of the `FPair` class representing the global minimum of
        the function. Default is `None`.
        """
        self.calls = 0
        self.limits = limits
        self.globmin = global_min

    @classmethod
    def __subclasshook__(cls, subclass):
        """
        Checks if a class is a subclass of ObjectiveFunction.

        #### Parameters
        **subclass**
        : The class to check.

        #### Notes
        This method checks if the provided class implements the `__call__`,
        `singlepoint`, `multipoint`, and `__repr__` methods. If it does, the
        class is considered a subclass of `ObjectiveFunction`.
        """
        return (
            hasattr(subclass, "__call__")
            and callable(subclass.__call__)
            and hasattr(subclass, "pointwise")
            and callable(subclass.pointwise)
            and hasattr(subclass, "multipoint")
            and callable(subclass.multipoint)
            and hasattr(subclass, "__repr__")
            and callable(subclass.__repr__)
            or NotImplemented
        )

    def __call__(self, pos):
        """
        Evaluates the function.

        #### Parameters
        **pos** (`npt.NDArray`)
        : The position(s) at which to evaluate the function.

        #### Notes
        This method checks the shape of the input and decides whether to call
        the `singlepoint` or `multipoint` method. It increments the `calls`
        attribute each time the function is evaluated.
        TODO: calls in multipoint may be more than once
        """
        if pos.ravel().shape[0] != self.limits.dims:
            self.calls += 1
            return self.multipoint(pos)
        else:
            self.calls += 1
            return self.singlepoint(pos)

    @abc.abstractmethod
    def singlepoint(self, pos):
        """
        Evaluates the function at a single point.

        #### Parameters
        **pos** (`npt.NDArray`)
        : The position at which to evaluate the function.

        #### Notes
        This method must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Need to be able to call the objective function on a single point"
        )

    @abc.abstractmethod
    def multipoint(self, pos):
        """
        Evaluates the function at multiple points.

        #### Parameters
        **pos** (`npt.NDArray`)
        : The positions at which to evaluate the function.

        #### Notes
        This method must be implemented by subclasses. It allows for a faster
        implementation in C.

        TODO: This allows for a faster implementation in C
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        """
        Returns a string representation of the function.

        #### Notes
        This method must be implemented by subclasses. It should return a string
        that names the function.
        """
        raise NotImplementedError
