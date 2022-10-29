import numpy as np 
from src.model import Model
from src.helpers import load_csv_data
from src.utils import cross_validation_visualization 
from src.data_processing import (
    clean_data,
    normalize_data,
    build_poly
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


def cross_validation(y, x, k_indices, k, lambda_, degree):
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

    # Form data with polynomial degree:
    tx_te = build_poly(x_te, degree)
    tx_tr = build_poly(x_tr, degree)

    # Train model:
    np.random.seed(1)
    model = Model(np.random.uniform(-1, 1, size=tx_tr.shape[1]), max_iters=10, gamma=0.1, lambda_=lambda_)
    model.train(y_tr, tx_tr, y_te, tx_te)
    
    # Return loss for train and test data: 
    return model.loss_tr[-1], model.loss_te[-1]


def best_params_selection(y, x, degrees, k_fold, lambdas, seed = 1):
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
                loss_tr, loss_te = cross_validation(y, x, k_indices, k, lambda_, degree)
                loss_tr_k.append(loss_tr)
                loss_te_k.append(loss_te)
            
            final_loss_tr[i,j] = np.mean(loss_tr_k)
            final_loss_te[i,j] = np.mean(loss_te_k)
    
    min_idx = np.unravel_index(final_loss_te.argmin(), final_loss_te.shape)
    best_degree = degrees[min_idx[0]]
    best_lambda = lambdas[min_idx[1]]
    best_loss = np.min(final_loss_te) 
    
    #cross_validation_visualization(lambdas, degrees, final_loss_te)
    
    return best_degree, best_lambda, best_loss


if __name__  == '__main__':
    # Load and prepare data for training:
    y, x, ids_train = load_csv_data(DATA_PATH + "train.csv", sub_sample=False)
    y[y == -1] = 0

    x = clean_data(x, mod="mean")
    x = normalize_data(x)
    
    degrees =  np.arange(1,3)
    lambdas = np.logspace(-4, 0, 3)
    best_degree, best_lambda, best_rmse = best_params_selection(y, x, degrees=degrees, lambdas=lambdas, k_fold=4)
    print("The best test loss of %.3f is obtained for a degree of %.f and a lambda of %.5f." % (best_rmse, best_degree, best_lambda))


# def best_lambda_selection(y, x, degree, k_fold, lambdas):
#     """cross validation over regularisation parameter lambda.
    
#     Args:
#         degree: integer, degree of the polynomial expansion
#         k_fold: integer, the number of folds
#         lambdas: shape = (p, ) where p is the number of values of lambda to test
#     Returns:
#         best_lambda : scalar, value of the best lambda
#         best_loss : scalar, the associated Log-loss for the best lambda
#     """
    
#     seed = 12
    
#     # Split data in k fold
#     k_indices = build_k_indices(y, k_fold, seed)
    
#     # Define lists to store the final log-loss of training data and test data
#     final_loss_tr = []
#     final_loss_te = []
    
#     # cross validation over lambdas:
#     for lambda_ in lambdas:
        
#         loss_tr_k = []
#         loss_te_k = []
#         for k in range(k_fold):
#             loss_tr, loss_te = cross_validation(y, x, k_indices, k, lambda_, degree)
#             loss_tr_k.append(loss_tr)
#             loss_te_k.append(loss_te)
        
#         final_loss_tr.append(np.mean(loss_tr_k))
#         final_loss_te.append(np.mean(loss_te_k))
    
#     best_lambda = lambdas[np.argmin(final_loss_te)]
#     best_loss = np.min(final_loss_te)
    
#     cross_validation_visualization(lambdas, final_loss_tr, final_loss_te)
#     print("For polynomial expansion up to degree %.f, the choice of lambda which leads to the best test loss is %.5f with a test loss of %.3f" % (degree, best_lambda, best_loss))
#     return best_lambda, best_loss