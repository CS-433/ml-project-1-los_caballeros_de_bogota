import numpy as np
from src.utils import (
    sigmoid,
    compute_loss,
    compute_gradient,
    compute_accuracy,
    compute_f1_score,
)


class Model:
    def __init__(self, weights):
        self.weights = weights
        self.loss_tr = []
        self.loss_te = []
        self.acc_tr = []
        self.acc_te = []
        self.f1 = []

    def train(self, y_tr, x_tr, y_te, x_te, max_iters, gamma):
        """Train model with Logistic regression using GD
        Args:
        """

        self.loss_tr.append(compute_loss(y_tr, x_tr, self.weights, "log"))
        self.loss_te.append(compute_loss(y_te, x_te, self.weights, "log"))
        self.acc_tr.append(compute_accuracy(y_tr, self.predict(x_tr)))
        self.acc_te.append(compute_accuracy(y_te, self.predict(x_te)))
        self.f1.append(compute_f1_score(y_te, self.predict(x_te)))
        print("Epoch {}/{}: Training Loss {}".format(0, max_iters, self.loss_tr[-1]))

        for epoch in range(1, max_iters + 1):
            # compute gradient
            grad = compute_gradient(y_tr, x_tr, self.weights, "log")

            # update w through the stochastic gradient update
            self.weights = self.weights - gamma * grad

            # calculate loss, accuracy and f1 score
            self.loss_tr.append(compute_loss(y_tr, x_tr, self.weights, "log"))
            self.loss_te.append(compute_loss(y_te, x_te, self.weights, "log"))
            self.acc_tr.append(compute_accuracy(y_tr, self.predict(x_tr)))
            self.acc_te.append(compute_accuracy(y_te, self.predict(x_te)))
            self.f1.append(compute_f1_score(y_te, self.predict(x_te)))

            # Print progress:
            print(
                "Epoch {}/{}: Training Loss {}".format(
                    epoch, max_iters, self.loss_tr[-1]
                )
            )

    def predict(self, x):
        y_pred = np.empty_like(x @ self.weights)
        y_pred[sigmoid(x @ self.weights) > 0.5] = 1
        y_pred[sigmoid(x @ self.weights) <= 0.5] = 0
        return y_pred
