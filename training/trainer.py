import numpy as np
from copy import deepcopy
from typing import Tuple
from numpy import ndarray
from models.neural_network import NeuralNetwork
from optimizers.sgd import Optimizer


def permute_data(X: ndarray, y: ndarray) -> Tuple[ndarray, ndarray]:
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


class Trainer:
    """Handles the training loop."""

    def __init__(self, net: NeuralNetwork, optim: Optimizer):
        self.net = net
        self.optim = optim
        self.best_loss = 1e9
        setattr(self.optim, "net", self.net)

    def generate_batches(self, X: ndarray, y: ndarray, size: int = 32):
        assert X.shape[0] == y.shape[0]
        for i in range(0, X.shape[0], size):
            yield X[i:i+size], y[i:i+size]

    def fit(self,
            X_train: ndarray, y_train: ndarray,
            X_test: ndarray, y_test: ndarray,
            epochs: int = 100,
            eval_every: int = 10,
            batch_size: int = 32,
            seed: int = 1,
            restart: bool = True) -> None:

        np.random.seed(seed)

        if restart:
            for layer in self.net.layers:
                layer.first = True
            self.best_loss = 1e9

        best_model = None   # last finite checkpoint with the best val loss
        best_epoch = 0

        for epoch in range(epochs):

            X_train, y_train = permute_data(X_train, y_train)

            diverged = False
            for X_batch, y_batch in self.generate_batches(X_train, y_train, batch_size):
                batch_loss = self.net.train_batch(X_batch, y_batch)
                if not np.isfinite(batch_loss):
                    diverged = True
                    break
                self.optim.step()

            if (epoch + 1) % eval_every == 0 or diverged:
                if diverged:
                    loss = float("nan")
                else:
                    preds = self.net.forward(X_test)
                    loss = self.net.loss.forward(preds, y_test)

                if np.isfinite(loss) and loss < self.best_loss:
                    print(f"Epoch {epoch+1} — val loss: {loss:.3f}")
                    self.best_loss = loss
                    best_model = deepcopy(self.net)
                    best_epoch = epoch + 1
                    continue

                if not np.isfinite(loss):
                    print(f"Epoch {epoch+1} — diverged (loss is nan/inf).")
                else:
                    print(f"Epoch {epoch+1} — loss increased.")

                if best_model is not None:
                    print(f"Restoring best model from epoch {best_epoch}")
                    self.net = best_model
                    setattr(self.optim, "net", self.net)
                else:
                    print("No finite checkpoint exists — lower lr / check input scaling.")
                break