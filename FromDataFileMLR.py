import time                 # provides timing for benchmarks
import sys
import csv
import numpy as np
from numpy import array, float64, sqrt
from sklearn import svm	    # provides Support Vector Regression

class DataFromFile:
    #------------------------------------------------------------------------------
    def getTwoDecPoint(self, x):
        return float("%.2f"%x)
    #------------------------------------------------------------------------------
    def placeDataIntoArray(self, fileName):
        try:
            return np.loadtxt(fileName, delimiter=',', dtype=float64)
        except Exception as e:
            print(f"Error reading file {fileName}: {e}")
            sys.exit(1)
    #------------------------------------------------------------------------------
    def getAllOfTheData(self):
        TrainX = self.placeDataIntoArray('Train-Data.csv')
        TrainY = self.placeDataIntoArray('Train-pIC50.csv')
        ValidateX = self.placeDataIntoArray('Validation-Data.csv')
        ValidateY = self.placeDataIntoArray('Validation-pIC50.csv')
        TestX = self.placeDataIntoArray('Test-Data.csv')
        TestY = self.placeDataIntoArray('Test-pIC50.csv')
        return TrainX, TrainY, ValidateX, ValidateY, TestX, TestY
    #------------------------------------------------------------------------------
    def rescaleTheData(self, TrainX, ValidateX, TestX):
        # Handle division by zero in variance
        TrainXVar = TrainX.var(axis=0, ddof=1)
        TrainXVar[TrainXVar == 0] = 1e-8  # Avoid division by zero
        TrainXMean = TrainX.mean(axis=0)

        TrainX = (TrainX - TrainXMean) / np.sqrt(TrainXVar)
        ValidateX = (ValidateX - TrainXMean) / np.sqrt(TrainXVar)
        TestX = (TestX - TrainXMean) / np.sqrt(TrainXVar)

        return TrainX, ValidateX, TestX
    #------------------------------------------------------------------------------
