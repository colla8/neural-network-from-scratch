import numpy as np
from numpy import ndarray
from core.layer import Layer
from core.operation import WeightMultiply, BiasAdd, Operation
from activations.sigmoid import Sigmoid
from activations.relu import ReLU


class Dense(Layer):
    """Fully connected layer."""

    def __init__(self, neurons: int, activation: Operation = None):
        super().__init__(neurons)
        self.activation = activation or Sigmoid()

    def _setup_layer(self, input_: ndarray) -> None:
        seed = getattr(self, "seed", None)
        if seed:
            np.random.seed(seed)

        n_in = input_.shape[1]

        # He init for ReLU, Xavier-style for the others
        if isinstance(self.activation, ReLU):
            scale = np.sqrt(2.0 / n_in)
        else:
            scale = np.sqrt(1.0 / n_in)

        self.params = [
            np.random.randn(n_in, self.neurons) * scale,   # weights
            np.zeros((1, self.neurons)),                    # bias
        ]

        self.operations = [
            WeightMultiply(self.params[0]),
            BiasAdd(self.params[1]),
            self.activation,
        ]