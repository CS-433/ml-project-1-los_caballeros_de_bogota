import pickle
import numpy as np
from src.model import Model
from src.utils import plot_performance
from src.helpers import load_csv_data, create_csv_submission
from src.data_processing import (
    filter_data,
    split_data
)

DATA_PATH = "data/"

if __name__ == "__main__":

    # Load and prepare data for training:
    y, x, ids= load_csv_data(DATA_PATH + "train.csv", sub_sample=True)
    y[y == -1] = 0
    
    data = filter_data(y, x, ids)
    models = {'no_mass': {'jet0': [], 
                    'jet1': [], 
                    'jet2': [], 
                    'jet3': []},
        'mass': {'jet0': [], 
                'jet1': [], 
                'jet2': [], 
                'jet3': []}}
    
    # with open('src/best_params_2.pkl', 'rb') as f:
    #     params = pickle.load(f)
    
    params = {'no_mass': {'jet0': {'degree':2, 'lambda_':0}, 
                'jet1': {'degree':2, 'lambda_':0}, 
                'jet2': {'degree':2, 'lambda_':0}, 
                'jet3': {'degree':2, 'lambda_':0}},
    'mass': {'jet0': {'degree':2, 'lambda_':0}, 
            'jet1': {'degree':2, 'lambda_':0}, 
            'jet2': {'degree':2, 'lambda_':0}, 
            'jet3': {'degree':2, 'lambda_':0}}}
        
    gammas = [1e-5, 1e-5, 1e-6, 1e-6, 1e-5, 1e-6, 1e-7, 1e-7]
    i = 0
    for key1, value1 in data.items():
        for key2, value2 in value1.items():
            
            x = data[key1][key2]['x']
            y = data[key1][key2]['y']
            x_tr, x_te, y_tr, y_te = split_data(x, y, ratio=0.75)
            
            # Training:
            print('Train model for sample {}-{}:'.format(key1, key2))
            
            degree = params[key1][key2]['degree']
            lambda_ = params[key1][key2]['lambda_']
            
            model = Model(max_iters=1000, gamma=gammas[i], degree=degree, lambda_=lambda_)
            model.train(y_tr, x_tr, y_te, x_te)

            # Print performance:
            plot_performance(model)
            print(
                "Model achieves {:.2f} accuracy and {:.2f} F1 score on test set.\n".format(
                    model.acc_te[-1], model.f1[-1]
                )
            )
            models[key1][key2] = model
            i += 1

    # # Create submission:
    # y, x, ids = load_csv_data(DATA_PATH + "test.csv", sub_sample=False)
    # data = filter_data(y, x, ids)
    # y_pred = np.array([])
    # ids_pred = np.array([])
    
    # for key1, value1 in data.items():
    #     for key2, value2 in value1.items():
            
    #         model = models[key1][key2]
    #         x = data[key1][key2]['x']
            
    #         y_pred = np.concatenate((y_pred, model.predict(x, eval_mode=True)))
    #         ids_pred = np.concatenate((ids_pred, data[key1][key2]['ids']))
    
    # y_pred[y_pred == 0] = -1
    # create_csv_submission(ids_pred, y_pred, DATA_PATH + "submission.csv")
    # print("Submission saved at 'data/submission.csv'.")
