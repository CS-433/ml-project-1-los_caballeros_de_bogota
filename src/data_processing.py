import numpy as np


def clean_data(x, mod):
    if mod == "mean":
        mask = x == -999.0
        x[mask] = 0
        means = np.mean(x, axis=0, where=np.logical_not(mask))
        return x + np.tile(means, (x.shape[0], 1)) * mask
    
    if mod == "med":
        mask = x == -999.0
        x[mask] = 0
        medians = np.median(x[np.logical_not(mask)], axis=0)
        return x + np.tile(medians, (x.shape[0], 1)) * mask


def normalize_data(x):
    normalized = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    return normalized


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.

    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.
    """
    # set seed
    np.random.seed(seed)

    idx = np.random.permutation(np.arange(len(x)))
    idx_max = np.floor(ratio * len(x)).astype(int)

    x_tr = x[idx][:idx_max]
    x_te = x[idx][idx_max:]
    y_tr = y[idx][:idx_max]
    y_te = y[idx][idx_max:]

    return x_tr, x_te, y_tr, y_te


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,D), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N,D*(degree+1))
    """

    poly = np.tile(x, (1, degree + 1))
    poly = poly ** np.repeat(np.arange(degree + 1), x.shape[1])

    return poly
