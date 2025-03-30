import time  # provides timing for benchmarks
from numpy import *  # provides complex math and array functions
from sklearn import svm  # provides Support Vector Regression
from sklearn import linear_model
from sklearn import neural_network
from sklearn import cross_decomposition  #provides Partial Least Square Regression
import csv
import math
import sys
import numpy as np
from FromDataFileMLR import DataFromFile

# Local files created by me
import mlr
import FromDataFileMLR
import FromFinessFileMLR

def calculate_fitness(predicted, actual):
    # Calculate R² and fitness
    residual = actual - predicted
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    fitness = ss_res  # Fitness is based on residual sum of squares
    return r2, fitness

def train_model(X, Y):
    # Placeholder for training logic (e.g., linear regression)
    coefficients = np.linalg.pinv(X.T @ X) @ X.T @ Y
    return coefficients

def predict(X, coefficients):
    return X @ coefficients

def genetic_algorithm():
    data_loader = DataFromFile()
    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = data_loader.getAllOfTheData()
    TrainX, ValidateX, TestX = data_loader.rescaleTheData(TrainX, ValidateX, TestX)

    population_size = 50
    num_features = TrainX.shape[1]
    population = np.random.choice([0, 1], size=(population_size, num_features), p=[0.985, 0.015])

    for generation in range(100):
        fitness_scores = []
        for individual in population:
            selected_features = np.where(individual == 1)[0]
            if len(selected_features) == 0:
                fitness_scores.append(float('inf'))
                continue

            X_train_selected = TrainX[:, selected_features]
            try:
                coefficients = train_model(X_train_selected, TrainY)
                predicted_train = predict(X_train_selected, coefficients)
                r2_train, fitness = calculate_fitness(predicted_train, TrainY)
                if r2_train < 0.5 or r2_train >= 1:
                    fitness = float('inf')
            except Exception as e:
                print(f"Error during training: {e}")
                fitness = float('inf')

            fitness_scores.append(fitness)

        sorted_indices = np.argsort(fitness_scores)
        top_individuals = population[sorted_indices[:2]]

        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = top_individuals
            crossover_point = np.random.randint(1, num_features - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            new_population.extend([child1, child2])

        mutation_rate = 0.01
        for individual in new_population:
            if np.random.rand() < mutation_rate:
                mutation_index = np.random.randint(num_features)
                individual[mutation_index] = 1 - individual[mutation_index]

        population = np.array(new_population)

    best_individual = population[sorted_indices[0]]
    selected_features = np.where(best_individual == 1)[0]
    print("Best features:", selected_features)

if __name__ == "__main__":
    genetic_algorithm()


