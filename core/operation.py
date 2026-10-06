import numpy as np
from numpy import ndarray


def assert_same_shape(array: ndarray, array_grad: ndarray) -> None:
    assert array.shape == array_grad.shape, \
        f"Shape mismatch: expected {array.shape}, got {array_grad.shape}"


class Operation:
    """Base class for all operations in the network."""

    def forward(self, input_: ndarray) -> ndarray:
        self.input_ = input_
        self.output = self._output()
        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        assert_same_shape(self.output, output_grad)
        self.input_grad = self._input_grad(output_grad)
        assert_same_shape(self.input_, self.input_grad)
        return self.input_grad

    def _output(self) -> ndarray:
        raise NotImplementedError

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        raise NotImplementedError


class ParamOperation(Operation):
    """Operation that has trainable parameters."""

    def __init__(self, param: ndarray):
        self.param = param

    def backward(self, output_grad: ndarray) -> ndarray:
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        assert_same_shape(self.param, self.param_grad)

        return self.input_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        raise NotImplementedError


class WeightMultiply(ParamOperation):
    """Matrix multiplication with weights."""

    def __init__(self, W: ndarray):
        super().__init__(W)

    def _output(self) -> ndarray:
        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        return np.dot(output_grad, self.param.T)

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        return np.dot(self.input_.T, output_grad)


class BiasAdd(ParamOperation):
    """Bias addition operation."""

    def __init__(self, B: ndarray):
        assert B.shape[0] == 1, "Bias must have shape (1, neurons)"
        super().__init__(B)

    def _output(self) -> ndarray:
        return self.input_ + self.param

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])
