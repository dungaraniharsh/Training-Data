import time                 # provides timing for benchmarks
import sys
import csv
import numpy as np
from numpy import array, float64, sqrt
from sklearn.svm import SVR  # provides Support Vector Regression

class DataFromFile:
    #------------------------------------------------------------------------------
    def getTwoDecPoint(self, x):
        """Round a number to two decimal points."""
        return float("%.2f" % x)
    #------------------------------------------------------------------------------
    def placeDataIntoArray(self, fileName):
        """Load data from a CSV file into a NumPy array."""
        try:
            return np.loadtxt(fileName, delimiter=',', dtype=float64)
        except Exception as e:
            print(f"Error reading file {fileName}: {e}")
            sys.exit(1)
    #------------------------------------------------------------------------------
    def getAllOfTheData(self):
        """Load training, validation, and test data from predefined CSV files."""
        start_time = time.time()  # Start timing
        TrainX = self.placeDataIntoArray('Train-Data.csv')
        TrainY = self.placeDataIntoArray('Train-pIC50.csv')
        ValidateX = self.placeDataIntoArray('Validation-Data.csv')
        ValidateY = self.placeDataIntoArray('Validation-pIC50.csv')
        TestX = self.placeDataIntoArray('Test-Data.csv')
        TestY = self.placeDataIntoArray('Test-pIC50.csv')
        elapsed_time = time.time() - start_time  # End timing
        print(f"Data loading completed in {elapsed_time:.2f} seconds.")
        return TrainX, TrainY, ValidateX, ValidateY, TestX, TestY
    #------------------------------------------------------------------------------
    def rescaleTheData(self, TrainX, ValidateX, TestX):
        """Rescale data using the mean and variance of the training set."""
        TrainXVar = np.clip(TrainX.var(axis=0, ddof=1), 1e-8, None)  # Avoid division by zero
        TrainXMean = TrainX.mean(axis=0)

        TrainX = (TrainX - TrainXMean) / np.sqrt(TrainXVar)
        ValidateX = (ValidateX - TrainXMean) / np.sqrt(TrainXVar)
        TestX = (TestX - TrainXMean) / np.sqrt(TrainXVar)

        return TrainX, ValidateX, TestX
    #------------------------------------------------------------------------------
    def trainAndPredictSVR(self, TrainX, TrainY, TestX):
        """Train a Support Vector Regression model and make predictions."""
        print("Training Support Vector Regression model...")
        start_time = time.time()  # Start timing
        model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        model.fit(TrainX, TrainY)
        elapsed_time = time.time() - start_time  # End timing
        print(f"Model training completed in {elapsed_time:.2f} seconds.")

        print("Making predictions on test data...")
        predictions = model.predict(TestX)
        return predictions
