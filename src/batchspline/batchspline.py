import numpy as np
from scipy.interpolate import UnivariateSpline


class BatchSpline:
    def __init__(self, b_spl: list[UnivariateSpline]):
        self.b_spl = b_spl

    def __call__(self, T_x: np.ndarray) -> np.ndarray:
        return np.stack([spl(T_x) for spl in self.b_spl], axis=-1)

    def derivative(self, n: int = 1) -> "BatchSpline":
        return BatchSpline([spl.derivative(n) for spl in self.b_spl])

    @staticmethod
    def get_spline(T_x: np.ndarray, T_y: np.ndarray, w: np.ndarray = None, k: int = 3, s: float = None):
        return get_spline(T_x, T_y, w=w, k=k, s=s)


def get_spline(T_x: np.ndarray, T_y: np.ndarray, w: np.ndarray = None, k: int = 3, s: float = None):
    assert len(T_x) == len(T_y)
    args = dict(w=w, k=k, s=s)

    if T_y.ndim == 1:
        return UnivariateSpline(T_x, T_y, **args)

    assert T_y.ndim == 2
    nx = T_y.shape[1]
    b_spl = [UnivariateSpline(T_x, T_y[:, ii], **args) for ii in range(nx)]

    return BatchSpline(b_spl)
