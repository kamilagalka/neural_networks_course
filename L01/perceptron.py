import logging

import numpy as np

logging.basicConfig(level=logging.INFO)


class Perceptron:
    def __init__(self, alpha, weights, use_bias=True):
        self.alpha = alpha
        self.weights = weights  # Usage assumption: bias weight / theta value as the last weight
        self.iteration_counter = 1
        self.use_bias = use_bias

    def _full_arousal(self, x):
        if self.use_bias:
            return np.dot(x, self.weights[:-1]) + self.weights[-1]

        return np.dot(x, self.weights[:-1])

    def _activation_func(self, z):
        if self.use_bias:
            return 1 if z > 0 else 0
        return 1 if z > self.weights[-1] else 0

    @staticmethod
    def _get_error(expected, actual):
        return expected - actual

    def _adjust_weights(self, error, input_data):
        self.weights[:-1] += self.alpha * error * input_data
        if self.use_bias:
            self.weights[-1] += self.alpha * error

    def train(self, train_input, train_output):
        no_errors = False
        while not no_errors:
            self.iteration_counter += 1
            no_errors = True

            for x, y in zip(train_input, train_output):
                z = self._full_arousal(x)
                perceptron_output = self._activation_func(z)
                perceptron_error = self._get_error(expected=y, actual=perceptron_output)
                self._adjust_weights(error=perceptron_error, input_data=x)
                if perceptron_error != 0:
                    no_errors = False

    def predict(self, input_vector):
        z = self._full_arousal(input_vector)
        return self._activation_func(z)


class PerceptronBipolar(Perceptron):
    def _activation_func(self, z):
        if self.use_bias:
            return 1 if z > 0 else -1
        return 1 if z > self.weights[-1] else -1
