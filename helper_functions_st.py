import numpy as np

## Support Vector Machine
def fit(X,Y):
    X = np.c_[np.ones((X.shape[0], 1)), X]
    w = np.zeros((X.shape[1],1))
    epochs = 1
    alpha = 0.001

    while (epochs < 10000):
        lambda_ = 1 / epochs
        for SampleIndex,SampleFeatures in enumerate(X):
            fx =  None



    y_pred = None
    return y_pred,w