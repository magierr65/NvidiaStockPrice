import matplotlib.pyplot as plt

def visualization_plot(y_true, y_pred, symbol):
    """
    params:
        y_true: 1D array
        y_pred: 1D array
        symbol: str
    return:
        None
    """
    plt.figure(figsize = (10,10))
    plt.plot(y_true.index, y_true.values, label="Actual price")
    plt.plot(y_true.index, y_pred, label="Predicted price")
    plt.legend()
    plt.title(f"{symbol} Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()