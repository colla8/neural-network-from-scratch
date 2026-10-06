import numpy as np
from numpy import ndarray
from models.neural_network import NeuralNetwork


def mae(y_true: ndarray, y_pred: ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: ndarray, y_pred: ndarray) -> float:
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))


def eval_regression_model(model: NeuralNetwork, X_test: ndarray, y_test: ndarray) -> None:
    preds = model.forward(X_test).reshape(-1, 1)
    print(f"MAE:  {mae(y_test, preds):.2f}")
    print(f"RMSE: {rmse(y_test, preds):.2f}")
