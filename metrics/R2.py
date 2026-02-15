import numpy as np

def R2_value(y_true, y_pred):
    """
    params:
        y_true: 1D array
        y_pred: 1D array
    returns:
        R2_value
    """
    mean = np.mean(y_true)
    R2_value = 1 - ( np.sum( (y_true - y_pred)**2 ) / np.sum( (y_true - mean)**2) )
    return R2_value