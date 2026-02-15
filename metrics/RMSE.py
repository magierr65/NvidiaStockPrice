import numpy as np

def RMSE_value(y_true, y_pred):
    """
    params:
        y_true: 1D array
        y_pred: 1D array
    returns:
        RMSE_value
    """
    n = len(y_true)
    RMSE_value = np.sqrt( 1/n * np.sum( ( y_true - y_pred )**2 ) )
    return RMSE_value
