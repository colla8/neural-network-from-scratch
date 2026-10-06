import numpy as np
from numpy import ndarray
from core.operation import Operation


class Sigmoid(Operation):
    """Sigmoid activation function."""

    def _output(self) -> ndarray:
        return 1.0 / (1.0 + np.exp(-self.input_))

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        sigmoid_backward = self.output * (1.0 - self.output)
        return sigmoid_backward * output_grad
