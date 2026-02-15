import numpy as np

def MAPE_value(y_true, y_pred):
    """
    params:
        y_true: 1D array
        y_pred: 1D array
    returns:
        MAPE_value
    """
    n = len(y_true)
    MAPE_value = 1 / n  * np.sum( np.abs( ( y_true - y_pred ) / y_true) ) * 100
    return MAPE_value
