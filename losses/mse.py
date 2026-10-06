import numpy as np
from numpy import ndarray
from core.operation import assert_same_shape


class Loss:
    """Base class for all loss functions."""

    def forward(self, prediction: ndarray, target: ndarray) -> float:
        assert_same_shape(prediction, target)
        self.prediction = prediction
        self.target = target
        return self._output()

    def backward(self) -> ndarray:
        self.input_grad = self._input_grad()
        assert_same_shape(self.prediction, self.input_grad)
        return self.input_grad

    def _output(self) -> float:
        raise NotImplementedError

    def _input_grad(self) -> ndarray:
        raise NotImplementedError


class MeanSquaredError(Loss):
    """Mean squared error loss."""

    def _output(self) -> float:
        return np.sum(np.power(self.prediction - self.target, 2)) / self.prediction.shape[0]

    def _input_grad(self) -> ndarray:
        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]
