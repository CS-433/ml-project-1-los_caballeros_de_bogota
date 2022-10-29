import numpy as np


def filter_data(y, x, ids):
    data = {'no_mass': {'jet0': {'x':[], 'y':[], 'ids':[]}, 
                        'jet1': {'x':[], 'y':[], 'ids':[]}, 
                        'jet2': {'x':[], 'y':[], 'ids':[]}, 
                        'jet3': {'x':[], 'y':[], 'ids':[]}},
            'mass': {'jet0': {'x':[], 'y':[], 'ids':[]}, 
                    'jet1': {'x':[], 'y':[], 'ids':[]}, 
                    'jet2': {'x':[], 'y':[], 'ids':[]}, 
                    'jet3': {'x':[], 'y':[], 'ids':[]}}}
    
    mask = x[:,0] == -999.0
    x_no_mass = x[mask,:]
    y_no_mass = y[mask]
    ids_no_mass = ids[mask]
    x_mass = x[np.logical_not(mask),:]
    y_mass = y[np.logical_not(mask)]
    ids_mass = ids[np.logical_not(mask)]
    
    data['mass']['jet0']['x'] = x_mass[x_mass[:,22]==0,:]
    data['mass']['jet0']['y'] = y_mass[x_mass[:,22]==0]
    data['mass']['jet0']['ids'] = ids_mass[x_mass[:,22]==0]
    data['no_mass']['jet0']['x'] = x_no_mass[x_no_mass[:,22]==0,:]
    data['no_mass']['jet0']['y'] = y_no_mass[x_no_mass[:,22]==0]
    data['no_mass']['jet0']['ids'] = ids_no_mass[x_no_mass[:,22]==0]
    data['mass']['jet0']['x'] = np.delete(data['mass']['jet0']['x'], [4,5,6,12,22,23,24,25,26,27,28,29], axis=1)
    data['no_mass']['jet0']['x'] = np.delete(data['no_mass']['jet0']['x'], [0,4,5,6,12,22,23,24,25,26,27,28,29], axis=1)
    
    data['mass']['jet1']['x'] = x_mass[x_mass[:,22]==1,:]
    data['mass']['jet1']['y'] = y_mass[x_mass[:,22]==1]
    data['mass']['jet1']['ids'] = ids_mass[x_mass[:,22]==1]
    data['no_mass']['jet1']['x'] = x_no_mass[x_no_mass[:,22]==1,:]
    data['no_mass']['jet1']['y'] = y_no_mass[x_no_mass[:,22]==1]
    data['no_mass']['jet1']['ids'] = ids_no_mass[x_no_mass[:,22]==1]
    data['mass']['jet1']['x'] = np.delete(data['mass']['jet1']['x'], [4,5,6,12,22,26,27,28], axis=1)
    data['no_mass']['jet1']['x'] = np.delete(data['no_mass']['jet1']['x'], [0,4,5,6,12,22,26,27,28], axis=1)
    
    data['mass']['jet2']['x'] = x_mass[x_mass[:,22]==2,:]
    data['mass']['jet2']['y'] = y_mass[x_mass[:,22]==2]
    data['mass']['jet2']['ids'] = ids_mass[x_mass[:,22]==2]
    data['no_mass']['jet2']['x'] = x_no_mass[x_no_mass[:,22]==2,:]
    data['no_mass']['jet2']['y'] = y_no_mass[x_no_mass[:,22]==2]
    data['no_mass']['jet2']['ids'] = ids_no_mass[x_no_mass[:,22]==2]
    data['mass']['jet2']['x'] = np.delete(data['mass']['jet2']['x'], [22], axis=1)
    data['no_mass']['jet2']['x'] = np.delete(data['no_mass']['jet2']['x'], [0,22], axis=1)
    
    data['mass']['jet3']['x'] = x_mass[x_mass[:,22]==3,:]
    data['mass']['jet3']['y'] = y_mass[x_mass[:,22]==3]
    data['mass']['jet3']['ids'] = ids_mass[x_mass[:,22]==3]
    data['no_mass']['jet3']['x'] = x_no_mass[x_no_mass[:,22]==3,:]
    data['no_mass']['jet3']['y'] = y_no_mass[x_no_mass[:,22]==3]
    data['no_mass']['jet3']['ids'] = ids_no_mass[x_no_mass[:,22]==3]
    data['mass']['jet3']['x'] = np.delete(data['mass']['jet3']['x'], [22], axis=1)
    data['no_mass']['jet3']['x'] = np.delete(data['no_mass']['jet3']['x'], [0,22], axis=1)
    
    return data


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
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    normalized = (x - mean) / std
    return normalized, mean, std


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
