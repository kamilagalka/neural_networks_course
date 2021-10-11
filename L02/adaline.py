import logging

import numpy as np

logging.basicConfig(level=logging.INFO)


class Adaline:
    def __init__(self, mi, weights, epsilon):
        self.mi = mi
        self.weights = weights  # Usage assumption: bias weight / theta value as the last weight
        self.iteration_counter = 0
        self.desired_epsilon = epsilon

    def _full_arousal(self, x):
        return np.dot(x, self.weights[:-1]) + self.weights[-1]

    def _activation_func(self, z):
        return 1 if z > 0 else -1

    @staticmethod
    def _get_error(expected, actual):
        return expected - actual

    def _adjust_weights(self, delta, input_data):
        self.weights[:-1] += self.mi * delta * input_data
        self.weights[-1] += self.mi * delta

    def train(self, train_input, train_output):
        L = len(train_input)
        epsilon = self.desired_epsilon + 1
        while epsilon > self.desired_epsilon:
            self.iteration_counter += 1
            deltas = 0

            for x, d in zip(train_input, train_output):
                delta_sqrt = (d - self._full_arousal(x))
                deltas += delta_sqrt**2
                self._adjust_weights(delta_sqrt, x)

            epsilon = deltas / L

            logging.info(epsilon)

    def predict(self, input_vector):
        z = self._full_arousal(input_vector)
        return self._activation_func(z)
