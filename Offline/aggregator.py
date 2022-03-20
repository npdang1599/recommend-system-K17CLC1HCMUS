import warnings
from sklearn.metrics import mean_squared_error
import numpy as np

class Aggregators:
    def __init__(self):
        pass
    
    #pass ratings or factors as input
    @staticmethod
    def average(arr):
        return np.average(arr, axis = 0, weights = None)

    @staticmethod
    def average_bf(arr):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            arr[arr == 0] = np.nan
            return np.nanmean(arr, axis=0)
    
    @staticmethod
    def weighted_average(arr, weights):
        return np.average(arr, axis = 0, weights = weights)