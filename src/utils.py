import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def compute_f1_score(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_pred == y_true))
    fp = np.sum((y_pred == 1) & (y_pred != y_true))
    fn = np.sum((y_pred == 0) & (y_pred != y_true))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return np.mean(e**2) / 2


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def calculate_logloss(y_true, y_pred, eps=1e-8):
    """Calculate the logloss"""
    return -np.mean(
        y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
    )


def compute_loss(y, tx, w, loss_type):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.
        loss_type: string in ["mae", "mse", "log"] specifying the type of loss to compute

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    e = y - tx @ w

    if loss_type == "mse":
        return calculate_mse(e)

    elif loss_type == "mae":
        return calculate_mae(e)

    elif loss_type == "log":
        y_pred = sigmoid(tx @ w)
        return calculate_logloss(y, y_pred)

    else:
        raise ValueError(
            "Invalid value for argument 'loss_type' when calling compute_loss, 'type' must be in ['mse', 'mae', 'log']."
        )


def compute_gradient(y, tx, w, loss_type, lambda_=0):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D, ). The vector of model parameters.
        loss_type: string in ["mse", "log"] specifying the type of loss

    Returns:
        An numpy array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """

    if loss_type == "mse":
        e = y - tx @ w
        grad = -(tx.T @ e) / y.shape[0]

    elif loss_type == "log":
        e = sigmoid(tx @ w) - y
        grad = (tx.T @ e) / y.shape[0]
        grad = grad + 2 * lambda_ * w
    return grad


def plot_performance(model):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(model.loss_tr)
    ax1.plot(model.loss_te)
    ax1.set(xlabel="Epochs", ylabel="Log-loss")
    ax1.legend(["Training", "Testing"])
    ax1.axes.get_xaxis().set_visible(False)
    ax1.grid(True)

    ax2.plot(model.acc_tr)
    ax2.plot(model.acc_te)
    ax2.plot(model.f1)
    ax2.set(xlabel="Epochs", ylabel="[-]")
    ax2.legend(["Training accuracy", "Testing accuracy", "F1-score"])
    ax2.grid(True)
    plt.show()