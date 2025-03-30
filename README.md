# Bioactivity Potency Predictor  
## Genetic Algorithm-Optimized Multiple Linear Regression for pIC50 Prediction  

### Overview  
This project predicts **pIC50** values (a measure of drug potency) using **Multiple Linear Regression (MLR)** with **Genetic Algorithm (GA)-based feature selection**. The goal is to optimize molecular descriptor selection to improve model performance using R², Q², and predictive cross-validation metrics.  

### Features  
- **Data Preprocessing:** Standardizes and normalizes training, validation, and test datasets.  
- **MLR Implementation:** Uses NumPy's least squares method for regression modeling.  
- **Genetic Algorithm for Feature Selection:** Evolves feature subsets to maximize predictive performance.  
- **Model Validation & Metrics:** Computes R², Q², and **Leave-One-Out Cross-Validation (LOOCV)** for robust evaluation.  

### File Structure  
- `FromDataFileMLR.py` → Loads and normalizes dataset.  
- `FromFinessFileMLR.py` → Computes fitness metrics and validates models.  
- `mlr.py` → Implements Multiple Linear Regression.  
- `MainMLR.py` → Runs the Genetic Algorithm for feature selection and model optimization.  
- `data/` → Contains CSV files for training, validation, and testing.  

### Requirements  
- Python 3.x  
- NumPy  
- Scikit-learn  
- Pandas  

### Running the Project  
1. Clone the repository:  
   ```sh
   git clone https://github.com/yourusername/Bioactivity-Potency-Predictor.git
   cd Bioactivity-Potency-Predictor
