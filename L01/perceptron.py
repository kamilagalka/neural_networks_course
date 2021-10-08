import logging

import numpy as np

logging.basicConfig(level=logging.INFO)


class Perceptron:
    def __init__(self, alpha, weights):
        self.alpha = alpha
        self.weights = weights  # Usage assumption: bias weight as the last weight
        self.iteration_counter = 1

    def _activation_func(self, x):
        return np.dot(x, self.weights[:-1]) + self.weights[-1]

    @staticmethod
    def _get_perceptron_output(z):
        return 1 if z > 0 else 0

    @staticmethod
    def _get_error(expected, actual):
        return expected - actual

    def _adjust_weights(self, error, input_data):
        self.weights[:-1] += self.alpha * error * input_data
        self.weights[-1] += self.alpha * error

    def train(self, train_input, train_output):
        no_errors = False
        while not no_errors:
            self.iteration_counter += 1
            no_errors = True

            for x, y in zip(train_input, train_output):
                z = self._activation_func(x)
                perceptron_output = self._get_perceptron_output(z)
                perceptron_error = self._get_error(expected=y, actual=perceptron_output)
                self._adjust_weights(error=perceptron_error, input_data=x)
                if perceptron_error != 0:
                    no_errors = False

    def predict(self, input_vector):
        z = self._activation_func(input_vector)
        return self._get_perceptron_output(z)
