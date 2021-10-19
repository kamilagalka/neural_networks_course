import logging

import numpy as np
from numpy.lib.function_base import average
from math import floor


class MLP:
    def __init__(self, starting_neurons_count, layers_count, output_labels_count, learning_factor):
        self.neurons_counts_in_each_layer = [starting_neurons_count]
        for layer_id in range(layers_count):
            self.neurons_counts_in_each_layer.append(
                floor(average([self.neurons_counts_in_each_layer[-1], output_labels_count])))
        self.neurons_counts_in_each_layer.append(output_labels_count)

        weights_in_each_layer = []
        for layer_id in range(layers_count + 1):
            weights_in_each_layer.append(
                np.random.normal(size=(self.neurons_counts_in_each_layer[layer_id] + 1,  # + bias
                                       self.neurons_counts_in_each_layer[layer_id + 1])))
            # np.random.rand(self.neurons_counts_in_each_layer[layer_id] + 1,  # + bias
            #                    self.neurons_counts_in_each_layer[layer_id + 1]))

        self.layers_count = layers_count
        self.output_labels_count = output_labels_count
        self.weights_in_each_layer = weights_in_each_layer
        self.learning_factor = learning_factor

    def train(self, training_data, training_labels):
        predicted_labels = []
        for image, expected_output in zip(training_data[:2], training_labels[:2]):
            a = image.flatten()

            for current_layer_number in range(self.layers_count + 1):
                a = np.append(a, 1)  # bias
                Z = self._full_excitation(a, self.weights_in_each_layer[current_layer_number])
                a = list(map(self._activation_func_sigm, Z))

            predicted_labels = self._output_function(a)

        return predicted_labels

    @staticmethod
    def _output_function(Z):
        S = sum([np.exp(z) for z in Z])
        return [np.exp(z) / S for z in Z]

    @staticmethod
    def _activation_func_sigm(z):
        a = 1 / (1 + np.exp(-z))
        return a

    @staticmethod
    def _activation_func_tanh(z):
        a = 2 / (1 + np.exp(-2 * z))
        return a

    @staticmethod
    def _activation_func_relu(z):
        a = 0 if z < 0 else z
        return a

    @staticmethod
    def _full_excitation(X, weights):
        Z = np.dot(X, weights)
        return Z

    def predict(self):
        pass
