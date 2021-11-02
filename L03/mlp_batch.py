import logging

import numpy as np


def init_weights(loc, scale, size):
    return np.random.normal(loc, scale, size)


class Layer:
    def __init__(self, activation_func, activation_func_derivative, weights, biases):
        self.activation_func = activation_func
        self.activation_func_derivative = activation_func_derivative
        self.weights = weights
        self.biases = biases


class MLP:
    def __init__(self, layers, learning_factor):
        self.layers = layers
        self.learning_factor = learning_factor

    def train(self, training_data, training_labels, num_of_epoch=10, batch_size=7):
        batches = self.get_training_batches(training_data, training_labels, batch_size)

        for epoch_id in range(num_of_epoch):
            logging.info(f"Epoch: {epoch_id}")
            for batch, batch_expected_output in batches:
                batch_activations = []
                batch_errors = []

                for image, expected_output in zip(batch, batch_expected_output):
                    excitations = []
                    activations = [image]
                    a = image

                    # feedforward
                    for layer in self.layers:
                        excitation = self._full_excitation(a, layer.weights, layer.biases)
                        excitations.append(excitation)
                        a = layer.activation_func(excitation)
                        activations.append(a)

                    # calculate errors in each layer
                    output_error = a - expected_output
                    errors = [output_error]
                    last_error = output_error
                    for layer_id in range(len(self.layers) - 2, -1, -1):
                        last_error = np.dot(self.layers[layer_id + 1].weights, last_error)
                        activation_der = self.layers[layer_id].activation_func_derivative(excitations[layer_id])
                        activation_der = np.transpose(activation_der)
                        last_error = last_error * activation_der
                        errors.append(last_error)

                    errors.reverse()

                    batch_errors.append(errors)
                    batch_activations.append(activations)

                batch_errors = np.array(batch_errors)
                batch_activations = np.array(batch_activations)

                # backpropagation
                for layer_id, layer in enumerate(self.layers):
                    # logging.info(f"be shape {batch_errors.shape}")
                    batch_errors_reshaped = np.dstack(batch_errors[:, layer_id])[0]
                    # logging.info(batch_errors_reshaped.shape)
                    batch_activations_reshaped = np.dstack(batch_activations[:, layer_id])[0]
                    batch_activations_reshaped = np.transpose(batch_activations_reshaped)

                    batch_errors_sum = np.sum(batch_errors[:, layer_id], axis=0)

                    layer.weights = layer.weights - self.learning_factor / batch_size * np.transpose(
                        np.dot(batch_errors_reshaped, batch_activations_reshaped))
                    layer.biases = layer.biases - self.learning_factor / batch_size * batch_errors_sum

            s = 0
            for ba, ex in batches:
                for t, e in zip(ba, ex):
                    data = t
                    for layer in self.layers:
                        tot_stim = self._full_excitation(data, layer.weights, layer.biases)
                        data = layer.activation_func(tot_stim)
                    s = s + (self.softmax(data).tolist().index(max(self.softmax(data))) == e.tolist().index(max(e)))
            logging.info(f"ACC: {s / len(training_data)}")

    @staticmethod
    def softmax(Z):
        exponents = np.exp(Z)
        return exponents / np.sum(exponents)

    @staticmethod
    def _softmax_gradient(y, a):
        return -(y - a)

    @staticmethod
    def _cost(expected_output, actual_output):
        return expected_output - actual_output

    @staticmethod
    def _cost_derivative(expected_output, actual_output, a):
        return -(expected_output - actual_output) * (np.transpose(a))

    @staticmethod
    def activation_func_sigm(z):
        a = 1 / (1 + np.exp(-z))
        return a

    @staticmethod
    def activation_func_sigm_derivative(z):
        return (1 - MLP.activation_func_sigm(z)) * MLP.activation_func_sigm(z)

    @staticmethod
    def _activation_func_tanh(z):
        a = 2 / (1 + np.exp(-2 * z))
        return a

    @staticmethod
    def _activation_func_relu(z):
        a = 0 if z < 0 else z
        return a

    @staticmethod
    def _full_excitation(X, weights, biases):
        Z = np.dot(X, weights) + biases
        return Z

    def predict(self, image):
        data = image.flatten()
        for layer in self.layers:
            tot_stim = self._full_excitation(data, layer.weights, layer.biases)
            data = layer.activation_func(tot_stim)
        return self.softmax(data).tolist().index(max(self.softmax(data)))

    def get_training_batches(self, training_data, training_output, batch_size=10):
        res = []
        res_training_data = []
        res_training_outputs = []

        for matrix, output in zip(training_data, training_output):
            res_training_data.append(matrix.flatten() / 255)
            expected_labels_one_in_n = np.zeros(10)  # output labels count = 10
            expected_labels_one_in_n[output] = 1
            res_training_outputs.append(expected_labels_one_in_n)

        for i in range(0, len(res_training_data), batch_size):
            res.append((np.concatenate([res_training_data[i:i + batch_size]]),
                        np.concatenate([res_training_outputs[i:i + batch_size]])))

        return res
