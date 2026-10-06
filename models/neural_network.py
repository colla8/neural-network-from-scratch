from numpy import ndarray
from typing import List
from core.layer import Layer
from losses.mse import Loss


class NeuralNetwork:
    """A simple feedforward neural network."""

    def __init__(self, layers: List[Layer], loss: Loss, seed: int = None):
        self.layers = layers
        self.loss = loss
        self.seed = seed

        if seed:
            for layer in self.layers:
                setattr(layer, "seed", seed)

    def forward(self, x: ndarray) -> ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad: ndarray) -> None:
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train_batch(self, x_batch: ndarray, y_batch: ndarray) -> float:
        predictions = self.forward(x_batch)
        loss = self.loss.forward(predictions, y_batch)
        self.backward(self.loss.backward())
        return loss

    def params(self):
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        for layer in self.layers:
            yield from layer.param_grads
