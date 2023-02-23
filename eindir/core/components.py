from dataclasses import dataclass
from collections import namedtuple
from enum import Enum
import abc

import numpy as np
import numpy.typing as npt
import typing

from eindir.core.exceptions import OutOfBounds

@dataclass
class FPair:
    """Class for handing position value pairs"""

    pos: npt.NDArray
    val: float

    def EvalFunc(self, ObjFunc: callable):
        self.val = ObjFunc(self.pos)


@dataclass
class NumLimit:
    """Class for tracking function bounds

    Parameters
    ----------
    low: A list of values which the function must not be below
    high: A list of values which the function must not exceed
    slack: The amount by which bounds may be exceeded without an error
    dims: This should be the same as the length of the list
    """

    low: npt.NDArray
    high: npt.NDArray
    slack: float = 1e-6
    dims: int = 1

    def check(self, pos: npt.NDArray):
        if not (
            np.all(pos > self.low - self.slack)
            and np.all(pos < self.high + self.slack)
        ):
            raise OutOfBounds(
                f"{pos} is not within {self.slack} of {self.low} and {self.high}"
            )
        return

    def mkpoint(self) -> npt.NDArray:
        """Generate a random point

        TODO: Handle other constraints (undefined regions)
        """
        return np.random.default_rng().uniform(self.low, self.high, self.dims)

    def clip(self, point: npt.NDArray) -> npt.NDArray:
        """Clips values"""
        return np.clip(point, self.low, self.high)


class ObjectiveFunction(metaclass=abc.ABCMeta):
    def __init__(self, limits: NumLimit, global_min: FPair = None):
        self.calls = 0
        self.limits = limits
        self.globmin = global_min

    @classmethod
    def __subclasshook__(cls, subclass):
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
        self.calls += 1
        ## TODO: calls in multipoint may be more than once
        if pos.ravel().shape[0] != self.limits.dims:
            return self.multipoint(pos)
        else:
            return self.singlepoint(pos)

    @abc.abstractmethod
    def singlepoint(self, pos):
        """Evaluate the function at a single configuration"""
        raise NotImplementedError(
            "Need to be able to call the objective function on a single point"
        )

    @abc.abstractmethod
    def multipoint(self, pos):
        """Evaluate the function at many configurations

        TODO: This allows for a faster implementation in C
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        """Name the function"""
        raise NotImplementedError
