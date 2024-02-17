import numpy as np

from eindir.core.components import FPair


def test_fpair():
    def dummy_func(x):
        return np.sum(x**2)

    pos = np.array([2, 3])
    pair = FPair(pos=pos, val=0)
    pair.EvalFunc(dummy_func)
    assert pair.val == 13, "FPair.EvalFunc does not correctly evaluate the function."
