import numpy as np 
from src.model import Model
from src.data_processing import build_poly
from src.utils import cross_validation_visualization 


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
    """return the loss of ridge regression for a fold corresponding to k_indices
    
    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test loss
    
    """
    
    # Get k'th subgroup in test, others in train: 
    x_te = x[k_indices[k]]
    y_te = y[k_indices[k]]
    
    x_tr = np.delete(x, k_indices[k])
    y_tr = np.delete(y, k_indices[k])

    # Form data with polynomial degree:
    tx_te = build_poly(x_te, degree)
    tx_tr = build_poly(x_tr, degree)

    # Train model:
    np.random.seed(1)
    model = Model(np.random.uniform(-1, 1, size=tx_tr.shape[1]), max_iters=500, gamma=0.1, lambda_=lambda_)
    model.train(y_tr, tx_tr, y_te, tx_te)
    
    # Return loss for train and test data: 
    return model.loss_tr[-1], model.loss_te[-1]


def best_lambda_selection(y, x, degree, k_fold, lambdas, model):
    """cross validation over regularisation parameter lambda.
    
    Args:
        degree: integer, degree of the polynomial expansion
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_loss : scalar, the associated Log-loss for the best lambda
    """
    
    seed = 12
    degree = degree
    k_fold = k_fold
    lambdas = lambdas
    
    # Split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # cross validation over lambdas:
    for lambda_ in lambdas:
        
        loss_tr_k = []
        loss_te_k = []
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation(y, x, k_indices, k, lambda_, degree)
            loss_tr_k.append(loss_tr)
            loss_te_k.append(loss_te)
        
        rmse_tr.append(np.mean(loss_tr_k))
        rmse_te.append(np.mean(loss_te_k))
    
    best_lambda = lambdas[np.argmin(rmse_te)]
    best_rmse = np.min(rmse_te)
    
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    print("For polynomial expansion up to degree %.f, the choice of lambda which leads to the best test rmse is %.5f with a test rmse of %.3f" % (degree, best_lambda, best_rmse))
    return best_lambda, best_rmse


best_lambda, best_rmse = cross_validation_demo(7, 4, np.logspace(-4, 0, 30))


def best_params_selection(degrees, k_fold, lambdas, seed = 1):
    """cross validation over regularisation parameter lambda and degree.
    
    Args:
        degrees: shape = (d,), where d is the number of degrees to test 
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_degree : integer, value of the best degree
        best_lambda : scalar, value of the best lambda
        best_rmse : value of the rmse for the couple (best_degree, best_lambda)
        
    >>> best_degree_selection(np.arange(2,11), 4, np.logspace(-4, 0, 30))
    (7, 0.004520353656360241, 0.28957280566456634)
    """
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    rmse_tr = np.zeros((len(degrees), len(lambdas)))
    rmse_te = np.zeros((len(degrees), len(lambdas)))
    
    for i, degree in enumerate(degrees):
    # cross validation over lambdas and degrees:
        for j, lambda_ in enumerate(lambdas):
            
            loss_tr_k = []
            loss_te_k = []
            for k in range(k_fold):
                loss_tr, loss_te = cross_validation(y, x, k_indices, k, lambda_, degree)
                loss_tr_k.append(loss_tr)
                loss_te_k.append(loss_te)
            
            rmse_tr[i,j] = np.mean(loss_tr_k)
            rmse_te[i,j] = np.mean(loss_te_k)
    
    min_idx = np.unravel_index(rmse_te.argmin(), rmse_te.shape)
    best_degree = degrees[min_idx[0]]
    best_lambda = lambdas[min_idx[1]]
    best_rmse = np.min(rmse_te) 
    
    return best_degree, best_lambda, best_rmse

if __name__  == '__main__':
    