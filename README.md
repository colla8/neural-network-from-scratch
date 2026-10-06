# Neural Network from Scratch with NumPy

A small, modular feedforward neural network library built with **NumPy only** (no PyTorch / TensorFlow). It was written to understand how forward passes, backpropagation and gradient-based training actually work under the hood.

The included notebook trains three models on the California Housing dataset (regression) and compares them.

## Features

- Modular design: operations, layers, losses, optimizers and the training loop are separate classes
- `Dense` layer with manual `forward` and `backward`
- Activations: `ReLU`, `Sigmoid`, `Linear`
- Loss: Mean Squared Error
- Optimizer: SGD
- Trainer with mini-batches, data shuffling, periodic validation and best-model restore
- Metrics: MAE and RMSE

## Project structure

```
.
├── core/
│   ├── operation.py      # Operation, ParamOperation, WeightMultiply, BiasAdd
│   └── layer.py          # Layer base class
├── layers/
│   └── dense.py          # Dense layer (weights + bias + activation)
├── activations/          # ReLU, Sigmoid, Linear
├── losses/
│   └── mse.py            # Loss base class, MeanSquaredError
├── optimizers/
│   └── sgd.py            # Optimizer base class, SGD
├── models/
│   └── neural_network.py # NeuralNetwork: chains layers, runs forward/backward
├── training/
│   └── trainer.py        # Trainer: training loop
├── utils/
│   └── metrics.py        # MAE, RMSE, eval_regression_model
└── notebook.ipynb        # Experiments on California Housing
```

### How it fits together

- `Operation` is the smallest unit: it has `forward` and `backward`. `ParamOperation` adds a trainable parameter and its gradient (`WeightMultiply`, `BiasAdd`). Activations are plain `Operation`s.
- A `Layer` chains operations; `Dense` = `WeightMultiply` → `BiasAdd` → activation. Weights are created lazily on the first forward pass, so you only specify the number of neurons.
- `NeuralNetwork` chains layers and holds the loss. `Trainer` runs the loop and calls the optimizer after each batch.

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install numpy scikit-learn jupyter
```

`scikit-learn` is only needed for the dataset, scaling and train/test split in the notebook. The library itself needs only NumPy.

## Quick start

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.neural_network import NeuralNetwork
from layers.dense import Dense
from activations.relu import ReLU
from activations.linear import Linear
from losses.mse import MeanSquaredError
from optimizers.sgd import SGD
from training.trainer import Trainer
from utils.metrics import eval_regression_model

# Data
X, y = fetch_california_housing(return_X_y=True)
y = y.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = NeuralNetwork(
    layers=[
        Dense(neurons=13, activation=ReLU()),
        Dense(neurons=13, activation=ReLU()),
        Dense(neurons=1, activation=Linear()),
    ],
    loss=MeanSquaredError(),
    seed=42,
)

# Train
trainer = Trainer(model, SGD(lr=0.01))
trainer.fit(X_train, y_train, X_test, y_test, epochs=50, eval_every=10)

# Evaluate
eval_regression_model(model, X_test, y_test)
```

Run it from the repository root so the imports resolve, or open `notebook.ipynb`.

## Results

California Housing, 80/20 split, standardized features, SGD (lr = 0.01), 50 epochs, batch size 32.

| Model | Architecture | MAE | RMSE |
|---|---|---|---|
| 1. Linear regression (baseline) | 1 neuron, Linear | 0.53 | 0.74 |
| 2. One hidden layer | 13 neurons (Sigmoid) → 1 | 0.45 | 0.65 |
| 3. Deep network | 13 (ReLU) → 13 (ReLU) → 1 | 0.38 | 0.56 |

Error drops steadily as model capacity grows. Model 3 was still improving at epoch 50, so more epochs would likely help.

## Notes and limitations

- Regression only, with MSE and plain SGD (no momentum or Adam).
- Dense layers only; there are no convolutional layers.
- The test set is used both for early stopping and for the final metrics, so the reported numbers are slightly optimistic. Use a separate validation split for a stricter evaluation.
- Weights use He initialization for ReLU layers and a `1/sqrt(n_in)` scale otherwise. Without proper scaling, deep ReLU networks can diverge to NaN on unnormalized or outlier-heavy inputs.

## License

Add a license of your choice (for example MIT) before publishing.
