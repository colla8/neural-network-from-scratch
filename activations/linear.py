import numpy as np
from numpy import ndarray
from core.operation import Operation


class Linear(Operation):
    """Identity activation — passes input through unchanged."""

    def _output(self) -> ndarray:
        return self.input_

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad
