import pickle
import numpy as np
import matplotlib.pyplot as plt
from src.utils import plot_performance

DATA_PATH = "data/"

if __name__ == "__main__":

    # Load pre-trained best model:
    with open("src/best_models.pkl", "rb") as f:
        models = pickle.load(f)

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

    # Iterate through each sub-model:
    for key1, value1 in models.items():
        for key2, value2 in value1.items():
            model = models[key1][key2]
            # Plot performance during training:
            plot_performance(
                model,
                "Performance during training of sub-model {}-{}".format(key1, key2),
            )
            global_acc += model.acc_te[-1] * prop[i]
            i += 1

    print("The combined models achieves {:.4f} global accuracy.\n".format(global_acc))
