import pickle
import numpy as np 
from src.model import Model
from src.helpers import load_csv_data
from src.data_processing import (
    normalize_data,
    filter_data
)

DATA_PATH = "data/"

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.
    
    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
    """
    
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree, mean, std):
    """return the loss of regularized logistic regression for a fold corresponding to k_indices
    
    Args:
        y:          shape=(N,d)
        x:          shape=(N,d)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test loss
    
    """
    
    # Get k'th subgroup in test, others in train: 
    x_te = x[k_indices[k],:]
    y_te = y[k_indices[k]]
    
    x_tr = np.delete(x, k_indices[k], axis=0)
    y_tr = np.delete(y, k_indices[k], axis=0)
    
    # Train model:
    model = Model(max_iters=2, gamma=0.1, mean=mean, std=std, degree=degree, lambda_=lambda_)
    model.train(y_tr, x_tr, y_te, x_te)
    
    # Return loss for train and test data: 
    return model.loss_tr[-1], model.loss_te[-1]


def best_params_selection(y, x, degrees, k_fold, lambdas, mean, std, seed = 1):
    """cross validation over regularisation parameter lambda and degree.
    
    Args:
        degrees: shape = (d,), where d is the number of degrees to test 
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_degree : integer, value of the best degree
        best_lambda : scalar, value of the best lambda
        best_rmse : value of the rmse for the couple (best_degree, best_lambda)
    """
    
    # Split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # Define lists to store the loss of training data and test data
    final_loss_tr = np.zeros((len(degrees), len(lambdas)))
    final_loss_te = np.zeros((len(degrees), len(lambdas)))
    
    for i, degree in enumerate(degrees):
    # cross validation over lambdas and degrees:
        for j, lambda_ in enumerate(lambdas):
            
            loss_tr_k = []
            loss_te_k = []
            for k in range(k_fold):
                loss_tr, loss_te = cross_validation(y, x, k_indices, k, lambda_, degree, mean, std)
                loss_tr_k.append(loss_tr)
                loss_te_k.append(loss_te)
            
            final_loss_tr[i,j] = np.mean(loss_tr_k)
            final_loss_te[i,j] = np.mean(loss_te_k)
    
    min_idx = np.unravel_index(final_loss_te.argmin(), final_loss_te.shape)
    best_degree = degrees[min_idx[0]]
    best_lambda = lambdas[min_idx[1]]
    best_loss = np.min(final_loss_te) 
    
    return best_degree, best_lambda, best_loss


if __name__  == '__main__':
    # Load and prepare data for training:
    y, x, ids= load_csv_data(DATA_PATH + "train.csv", sub_sample=False)
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
    
    params = {'no_mass': {'jet0': {'degree':[], 'lambda_':[]}, 
                    'jet1': {'degree':[], 'lambda_':[]}, 
                    'jet2': {'degree':[], 'lambda_':[]}, 
                    'jet3': {'degree':[], 'lambda_':[]}},
        'mass': {'jet0': {'degree':[], 'lambda_':[]}, 
                'jet1': {'degree':[], 'lambda_':[]}, 
                'jet2': {'degree':[], 'lambda_':[]}, 
                'jet3': {'degree':[], 'lambda_':[]}}}
    
    for key1, value1 in data.items():
        for key2, value2 in value1.items():
            print('Cross-validation for model {}-{}:'.format(key1, key2))
            x = data[key1][key2]['x']
            y = data[key1][key2]['y']
            x, mean, std = normalize_data(x)
            
            degrees =  np.arange(1,3)
            lambdas = np.logspace(-4, 0, 3)
            best_degree, best_lambda, best_loss = best_params_selection(y, x, degrees=degrees, lambdas=lambdas, k_fold=4, mean=mean, std=std)
            print("The best test loss of %.3f is obtained for a degree of %.f and a lambda of %.5f.\n" % (best_loss, best_degree, best_lambda))
            
            params[key1][key2]['degree'] = best_degree
            params[key1][key2]['lambda_'] = best_lambda
    
    with open('src/best_params.pkl', 'wb') as f:
        pickle.dump(params, f)
    print("Best parameters saved at 'src/best_params.pkl'.")