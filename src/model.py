import numpy as np
from src.data_processing import build_poly
from src.utils import (
    sigmoid,
    compute_loss,
    compute_gradient,
    compute_accuracy,
    compute_f1_score,
)


class Model:
    def __init__(self, max_iters, gamma, mean, std, degree, lambda_=0.0):
        self.max_iters = max_iters
        self.gamma = gamma 
        self.mean = mean
        self.std = std
        self.degree = degree
        self.lambda_ = lambda_
        self.loss_tr = []
        self.loss_te = []
        self.acc_tr = []
        self.acc_te = []
        self.f1 = []

    def train(self, y_tr, x_tr, y_te, x_te):
        """Train model with regularized logistic regression using GD
        Args:
        """
        # Feature augmentation:
        x_tr = build_poly(x_tr, self.degree)
        x_te = build_poly(x_te, self.degree)
        
        # Initialize weights:
        np.random.seed(1)
        self.weights = np.random.uniform(-1, 1, size=x_tr.shape[1])
        
        # Initialize loss, accuracy and f1-score:
        self.loss_tr.append(compute_loss(y_tr, x_tr, self.weights, "log"))
        self.loss_te.append(compute_loss(y_te, x_te, self.weights, "log"))
        self.acc_tr.append(compute_accuracy(y_tr, self.predict(x_tr)))
        self.acc_te.append(compute_accuracy(y_te, self.predict(x_te)))
        self.f1.append(compute_f1_score(y_te, self.predict(x_te)))
        print("Epoch {}/{}: Training Loss {}".format(0, self.max_iters, self.loss_tr[-1]))

        for epoch in range(1, self.max_iters + 1):
            # compute gradient
            grad = compute_gradient(y_tr, x_tr, self.weights, "log", lambda_=self.lambda_)

            # update w through the stochastic gradient update
            self.weights = self.weights - self.gamma * grad

            # calculate loss, accuracy and f1 score
            self.loss_tr.append(compute_loss(y_tr, x_tr, self.weights, "log"))
            self.loss_te.append(compute_loss(y_te, x_te, self.weights, "log"))
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
        if eval_mode:
            x = (x - self.mean)/self.std
            x = build_poly(x, self.degree)
        
        y_pred = np.empty_like(x @ self.weights)
        y_pred[sigmoid(x @ self.weights) > 0.5] = 1
        y_pred[sigmoid(x @ self.weights) <= 0.5] = 0
        return y_pred
