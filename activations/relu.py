import numpy as np
from numpy import ndarray
from core.operation import Operation


class ReLU(Operation):
    """ReLU activation function."""

    def _output(self) -> ndarray:
        return np.maximum(0, self.input_)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad * (self.input_ > 0)
