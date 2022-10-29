import numpy as np
from src.model import Model
from src.utils import plot_performance
from src.helpers import load_csv_data, create_csv_submission
from src.data_processing import (
    clean_data,
    normalize_data,
    split_data,
    build_poly,
)

DATA_PATH = "data/"

if __name__ == "__main__":

    # Load and prepare data for training:
    y, x, ids_train = load_csv_data(DATA_PATH + "train.csv", sub_sample=False)
    y[y == -1] = 0

    x = clean_data(x, mod="mean")
    x = normalize_data(x)
    x_tr, x_te, y_tr, y_te = split_data(x, y, ratio=0.75)

    # Feature augmentation:
    tx_tr = build_poly(x_tr, 2)
    tx_te = build_poly(x_te, 2)

    # Training:
    np.random.seed(1)
    model = Model(np.random.uniform(-1, 1, size=tx_tr.shape[1]), max_iters=500, gamma=0.1, lambda_=0.1)
    model.train(y_tr, tx_tr, y_te, tx_te)

    # Plot performance:
    plot_performance(model)
    print(
        "Model achieves {:.2f} accuracy and {:.2f} F1 score on test set.".format(
            model.acc_te[-1], model.f1[-1]
        )
    )

    # Create submission:
    y_test, x_test, ids_test = load_csv_data(DATA_PATH + "test.csv", sub_sample=False)
    x_test = clean_data(x_test, mod="mean")
    x_test = normalize_data(x_test)
    tx_test = augment_features(x_test)
    
    y_pred = model.predict(tx_test)
    y_pred[y_pred == 0] = -1
    create_csv_submission(ids_test, y_pred, DATA_PATH + "submission.csv")
    print("Submission saved at 'data/submission.csv'.")
