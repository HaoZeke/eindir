import numpy as np

from eindir.core.components import NumLimit, ObjectiveFunction


class DummyObjectiveFunction(ObjectiveFunction):
    def singlepoint(self, pos):
        return np.sum(pos**2)

    def multipoint(self, pos):
        return np.array([self.singlepoint(p) for p in pos])

    def __repr__(self):
        return "DummyObjectiveFunction"


def test_objective_function():
    limits = NumLimit(low=np.array([-5, -5]), high=np.array([5, 5]))
    obj_func = DummyObjectiveFunction(limits=limits)
    single_point = np.array([1, 1])
    multi_point = np.array([[1, 1], [2, 2]])

    obj_func(single_point)
    assert obj_func(single_point)[0] == 1, "Single point evaluation failed."

    multi_result = obj_func(multi_point)
    expected_results = np.array([2, 8])
    assert np.allclose(multi_result, expected_results), "Multi-point evaluation failed."
