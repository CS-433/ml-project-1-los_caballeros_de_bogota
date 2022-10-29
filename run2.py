import numpy as np
from src.model import Model
from src.utils import plot_performance
from src.helpers import load_csv_data, create_csv_submission
from src.data_processing import (
    filter_data,
    normalize_data,
    split_data,
    build_poly
)

DATA_PATH = "data/"

if __name__ == "__main__":

    # Load and prepare data for training:
    y, x, ids_train = load_csv_data(DATA_PATH + "train.csv", sub_sample=False)
    y[y == -1] = 0
    
    data = filter_data(y, x)
    models = {'no_mass': {'jet0': [], 
                    'jet1': [], 
                    'jet2': [], 
                    'jet3': []},
        'mass': {'jet0': [], 
                'jet1': [], 
                'jet2': [], 
                'jet3': []}}
    
    for key1, value1 in data.items():
        for key2, value2 in value1.items():
            
            x = data[key1][key2]['x']
            y = data[key1][key2]['y']
            x, mean, std = normalize_data(x)
            x_tr, x_te, y_tr, y_te = split_data(x, y, ratio=0.75)

            # Feature augmentation:
            tx_tr = build_poly(x_tr, 2)
            tx_te = build_poly(x_te, 2)

            # Training:
            print('Train model for sample {}-{}:'.format(key1, key2))
            np.random.seed(1)
            model = Model(np.random.uniform(-1, 1, size=tx_tr.shape[1]), max_iters=100, gamma=0.1, mean=mean, std=std, lambda_=0.1)
            model.train(y_tr, tx_tr, y_te, tx_te)

            # Print performance:
            #plot_performance(model)
            print(
                "Model achieves {:.2f} accuracy and {:.2f} F1 score on test set.\n".format(
                    model.acc_te[-1], model.f1[-1]
                )
            )
            models[key1][key2] = model

    # # Create submission:
    y_test, x_test, ids_test = load_csv_data(DATA_PATH + "test.csv", sub_sample=False)
    data = filter_data(y_test, x_test)
    
    for key1, value1 in data.items():
        for key2, value2 in value1.items():
            
            model = models[key1][key2]
            x = data[key1][key2]['x']
            
            # Normalize data:
            x = (x - model.mean) / model.std
            
            tx = build_poly(x, 2)
            
            y_pred = model.predict(tx)

        # Feature augmentation:
        tx_tr = build_poly(x_tr, 2)
        tx_te = build_poly(x_te, 2)

        # Training:
        print('Train model for sample {}-{}:'.format(key1, key2))
        np.random.seed(1)
        model = Model(np.random.uniform(-1, 1, size=tx_tr.shape[1]), max_iters=100, gamma=0.1, mean=mean, std=std, lambda_=0.1)
        model.train(y_tr, tx_tr, y_te, tx_te)

        # Print performance:
        #plot_performance(model)
        print(
            "Model achieves {:.2f} accuracy and {:.2f} F1 score on test set.\n".format(
                model.acc_te[-1], model.f1[-1]
            )
        )
        models[key1][key2] = model
    

    y_pred[y_pred == 0] = -1
    create_csv_submission(ids_test, y_pred, DATA_PATH + "submission.csv")
    print("Submission saved at 'data/submission.csv'.")
