import numpy as np
from numpy.typing import NDArray


class Solution:
    learning_rate = 0.01

    def get_model_prediction(
        self,
        X: NDArray[np.float64],
        weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        return X @ weights 


    def get_derivative(
        self,
        model_prediction: NDArray[np.float64],
        ground_truth: NDArray[np.float64],
        X: NDArray[np.float64],
        desired_weight: int
    ) -> float:

        N = X.shape[0]

        error = ground_truth - model_prediction
        feature_column = X[:, desired_weight]

        return (-2 / N) * np.dot(error, feature_column)


    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        weights = initial_weights.copy()
        n_features = X.shape[1]
        N = X.shape[0]

        for _ in range(num_iterations):

            # 1. predict
            predictions = self.get_model_prediction(X, weights)

            # 2. update each weight
            for j in range(n_features):

                gradient = self.get_derivative(
                    predictions,
                    Y,
                    X,
                    j
                )

                weights[j] -= self.learning_rate * gradient

        return np.round(weights, 5)    
            

