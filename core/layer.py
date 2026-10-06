from numpy import ndarray
from typing import List
from core.operation import Operation, ParamOperation, assert_same_shape


class Layer:
    """Base class for all layers."""

    def __init__(self, neurons: int):
        self.neurons = neurons
        self.first = True
        self.params: List[ndarray] = []
        self.param_grads: List[ndarray] = []
        self.operations: List[Operation] = []

    def _setup_layer(self, input_: ndarray) -> None:
        raise NotImplementedError

    def forward(self, input_: ndarray) -> ndarray:
        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for operation in self.operations:
            input_ = operation.forward(input_)

        self.output = input_
        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        assert_same_shape(self.output, output_grad)

        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)

        self._param_grads()
        return output_grad

    def _param_grads(self) -> None:
        self.param_grads = [
            op.param_grad
            for op in self.operations
            if issubclass(op.__class__, ParamOperation)
        ]

    def _params(self) -> None:
        self.params = [
            op.param
            for op in self.operations
            if issubclass(op.__class__, ParamOperation)
        ]
