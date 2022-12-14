import pickle
import numpy as np
from src.model import Model
from src.helpers import load_csv_data, create_csv_submission
from src.data_processing import filter_data, split_data

DATA_PATH = "data/"

if __name__ == "__main__":

    # Load and prepare data for training:
    y, x, ids = load_csv_data(DATA_PATH + "train.csv", sub_sample=False)
    y[y == -1] = 0

    # Filter data by mass and jet num:
    data = filter_data(y, x, ids)
    models = {
        "no_mass": {"jet0": [], "jet1": [], "jet2": [], "jet3": []},
        "mass": {"jet0": [], "jet1": [], "jet2": [], "jet3": []},
    }

    # Load best parameters found after 4-fold cross-validation:
    with open("src/best_params.pkl", "rb") as f:
        params = pickle.load(f)

    # Define proportion of each sub-set of the dataset to compute global accuracy:
    prop = [
        0.104492,
        0.030248,
        0.011808,
        0.005908,
        0.29516,
        0.279928,
        0.189708,
        0.082748,
    ]
    global_acc = 0

    i = 0
    for key1, value1 in data.items():
        for key2, value2 in value1.items():

            x = data[key1][key2]["x"]
            y = data[key1][key2]["y"]
            # Split data to assess performance during training:
            x_tr, x_te, y_tr, y_te = split_data(x, y, ratio=0.99)

            # Training:
            print("Train model for sample {}-{}:".format(key1, key2))
            degree = params[key1][key2]["degree"]
            lambda_ = params[key1][key2]["lambda_"]
            model = Model(max_iters=15000, gamma=1e-1, degree=degree, lambda_=lambda_)
            model.train(y_tr, x_tr, y_te, x_te)

            # Print performance:
            print(
                "Model achieves {:.2f} accuracy and {:.2f} F1 score on test set.\n".format(
                    model.acc_te[-1], model.f1[-1]
                )
            )
            models[key1][key2] = model
            global_acc += model.acc_te[-1] * prop[i]
            i += 1

    print(
        "The combined models achieves {:.4f} global accuracy on test set.\n".format(
            global_acc
        )
    )

    # Create submission:
    y, x, ids = load_csv_data(DATA_PATH + "test.csv", sub_sample=False)
    data = filter_data(y, x, ids)
    y_pred = np.array([])
    ids_pred = np.array([])

    # Make prediction for each sub-set of the test data using the corresponding trained sub-model:
    for key1, value1 in data.items():
        for key2, value2 in value1.items():

            model = models[key1][key2]
            x = data[key1][key2]["x"]

            y_pred = np.concatenate((y_pred, model.predict(x, eval_mode=True)))
            ids_pred = np.concatenate((ids_pred, data[key1][key2]["ids"]))

    y_pred[y_pred == 0] = -1
    create_csv_submission(ids_pred, y_pred, DATA_PATH + "submission.csv")
    print("Submission saved at 'data/submission.csv'.")
