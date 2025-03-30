# """Multiple Linear Regression"""
import numpy as np      

class MLR:
    # """Multiple Linear Regression"""
    
    def __init__(self):
        # """Initialization"""
        self.coef = None
    
    def fit(self, x_set, y_set):
        if x_set.shape[0] != y_set.shape[0]:
            raise ValueError("Number of rows in X and Y must match.")
        # Add a column of 1's for the intercept
        x_set = np.append(np.ones((x_set.shape[0], 1)), x_set, axis=1)
        # Use np.linalg.lstsq for numerical stability and handle edge cases
        self.coef, _, _, _ = np.linalg.lstsq(x_set, y_set, rcond=None)
        return 'MLR'
    
    def predict(self, x_set):
        if self.coef is None:
            raise ValueError("Model must be fitted before prediction.")
        # Ensure input is reshaped correctly
        if len(x_set.shape) == 1:
            x_set = np.reshape(x_set, (1, x_set.shape[0]))
        # Add a column of 1's for the intercept
        x_set = np.append(np.ones((x_set.shape[0], 1)), x_set, axis=1)
        return np.dot(x_set, self.coef)

    def printing(self):
        # """Print a message for debugging purposes."""
        print("How are you doing?")
