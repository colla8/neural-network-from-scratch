class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent."""

    def __init__(self, lr: float = 0.01):
        super().__init__(lr)

    def step(self) -> None:
        for param, grad in zip(self.net.params(), self.net.param_grads()):
            param -= self.lr * grad
