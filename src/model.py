import numpy as np
from src.data_processing import build_poly, normalize_data
from src.utils import (
    sigmoid,
    compute_loss,
    compute_gradient,
    compute_accuracy,
    compute_f1_score,
)


class Model:
    def __init__(self, max_iters, gamma, degree, lambda_=0.0):
        self.max_iters = max_iters
        self.gamma = gamma
        self.degree = degree
        self.lambda_ = lambda_
        self.loss_tr = []
        self.loss_te = []
        self.acc_tr = []
        self.acc_te = []
        self.f1 = []

    def train(self, y_tr, x_tr, y_te, x_te):
        """Train model with regularized logistic regression using GD"""

        # Feature augmentation:
        n_feat = x_tr.shape[1]
        x_tr = build_poly(x_tr, self.degree)
        x_te = build_poly(x_te, self.degree)

        # Normalize data (do not normalize features of degree 0):
        x_tr[:, n_feat:], self.mean, self.std = normalize_data(x_tr[:, n_feat:])
        x_te[:, n_feat:] = (x_te[:, n_feat:] - self.mean) / self.std

        # Initialize weights:
        np.random.seed(1)
        self.weights = np.random.uniform(-1, 1, size=x_tr.shape[1])

        # Initialize loss, accuracy and f1-score:
        self.loss_tr.append(
            compute_loss(y_tr, x_tr, self.weights, "log")
            + self.lambda_ * np.sum(self.weights**2)
        )
        self.loss_te.append(
            compute_loss(y_te, x_te, self.weights, "log")
            + self.lambda_ * np.sum(self.weights**2)
        )
        self.acc_tr.append(compute_accuracy(y_tr, self.predict(x_tr)))
        self.acc_te.append(compute_accuracy(y_te, self.predict(x_te)))
        self.f1.append(compute_f1_score(y_te, self.predict(x_te)))
        print(
            "Epoch {}/{}: Training Loss {}".format(0, self.max_iters, self.loss_tr[-1])
        )

        for epoch in range(1, self.max_iters + 1):

            # compute gradient
            grad = compute_gradient(
                y_tr, x_tr, self.weights, "log", lambda_=self.lambda_
            )

            # update w through the stochastic gradient update
            self.weights = self.weights - self.gamma * grad

            # calculate loss, accuracy and f1 score
            self.loss_tr.append(
                compute_loss(y_tr, x_tr, self.weights, "log")
                + self.lambda_ * np.sum(self.weights**2)
            )
            self.loss_te.append(
                compute_loss(y_te, x_te, self.weights, "log")
                + self.lambda_ * np.sum(self.weights**2)
            )
            self.acc_tr.append(compute_accuracy(y_tr, self.predict(x_tr)))
            self.acc_te.append(compute_accuracy(y_te, self.predict(x_te)))
            self.f1.append(compute_f1_score(y_te, self.predict(x_te)))

            # Print progress:
            print(
                "Epoch {}/{}: Training Loss {}".format(
                    epoch, self.max_iters, self.loss_tr[-1]
                )
            )

    def predict(self, x, eval_mode=False):
        """Make prediction with the current weights"""

        if eval_mode:
            # If eval_mode, make polynomial expansion and normalize the sample:
            n_feat = x.shape[1]
            x = build_poly(x, self.degree)
            x[:, n_feat:] = (x[:, n_feat:] - self.mean) / self.std

        y_pred = np.empty_like(x @ self.weights)
        y_pred[sigmoid(x @ self.weights) > 0.5] = 1
        y_pred[sigmoid(x @ self.weights) <= 0.5] = 0
        return y_pred
