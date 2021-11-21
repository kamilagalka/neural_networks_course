import copy
import logging

import numpy as np


def init_weights(weights_range, size, weight_init_method):
    if weight_init_method == "":
        weights = np.random.randn(*size) * weights_range
        return weights

    if weight_init_method == "Xavier":
        weights = np.random.randn(*size) * np.sqrt(2 / (sum(size)))
        return weights

    if weight_init_method == "He":
        weights = np.random.randn(*size) * np.sqrt(2 / size[0])
        return weights


class Layer:
    def __init__(self, activation_func, activation_func_derivative, weights, biases):
        self.activation_func = activation_func
        self.activation_func_derivative = activation_func_derivative
        self.weights = weights
        self.biases = biases

        self.previous_weights_delta = np.zeros((weights.shape[0], weights.shape[1]))
        self.previous_biases_delta = np.zeros((weights.shape[1],))
        self.weight_accumulators = np.zeros((weights.shape[0], weights.shape[1]))
        self.bias_accumulators = np.zeros((weights.shape[1],))
        self.v_weight_accumulators = np.zeros((weights.shape[0], weights.shape[1]))
        self.v_bias_accumulators = np.zeros((weights.shape[1],))


class MLP:
    def __init__(self, layers, learning_factor):
        self.layers = layers
        self.learning_factor = learning_factor

    def train(self, training_data, training_labels, validation_data, validation_labels, num_of_epoch=4, batch_size=7,
              optimizer=None):
        batches = self.get_training_batches(training_data, training_labels, batch_size)
        validation_batches = self.get_training_batches(validation_data, validation_labels, batch_size)

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

                    weights_delta = np.transpose(
                        np.dot(batch_errors_reshaped, batch_activations_reshaped))

                    if not optimizer:
                        self.no_optimizer(layer=layer, weight_gradients=weights_delta, bias_gradients=batch_errors_sum,
                                          batch_size=batch_size)
                    elif optimizer == 'momentum':
                        self.momentum(layer=layer, weight_gradients=weights_delta, bias_gradients=batch_errors_sum,
                                      batch_size=batch_size, momentum=0.8)
                    elif optimizer == 'nesterov':
                        self.nesterov_momentum(layer=layer, weight_gradients=weights_delta,
                                               bias_gradients=batch_errors_sum,
                                               batch_size=batch_size, momentum=0.8)
                    elif optimizer == 'adagrad':
                        self.adagrad(layer=layer, weight_gradients=weights_delta, bias_gradients=batch_errors_sum,
                                     batch_size=batch_size)
                    elif optimizer == 'adadelta':
                        self.adadelta(layer=layer, weight_gradients=weights_delta, bias_gradients=batch_errors_sum,
                                      batch_size=batch_size)
                    elif optimizer == 'adam':
                        self.adam(layer=layer, weight_gradients=weights_delta, bias_gradients=batch_errors_sum,
                                  batch_size=batch_size)

            # save weights to csv
            # for layer_id, layer in enumerate(self.layers):
            #     weights_to_csv = np.asarray(layer.weights)
            #     biases_to_csv = np.asarray(layer.biases)
            #     np.savetxt(f"weights/weights_{layer_id}.csv", weights_to_csv, delimiter=",")
            #     np.savetxt(f"weights/biases_{layer_id}.csv", biases_to_csv, delimiter=",")

            # calc accuracy
            s = 0
            for ba, ex in validation_batches:
                for t, e in zip(ba, ex):
                    data = t
                    for layer in self.layers:
                        tot_stim = self._full_excitation(data, layer.weights, layer.biases)
                        data = layer.activation_func(tot_stim)
                    s = s + (self.softmax(data).tolist().index(max(self.softmax(data))) == e.tolist().index(max(e)))
            logging.info(f"ACC: {s / len(validation_data)}")

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
    def _full_excitation(X, weights, biases):
        Z = np.dot(X, weights) + biases
        return Z

    @staticmethod
    def activation_func_sigm(z):
        a = 1 / (1 + np.exp(-z))
        return a

    @staticmethod
    def activation_func_sigm_derivative(z):
        return (1 - MLP.activation_func_sigm(z)) * MLP.activation_func_sigm(z)

    @staticmethod
    def activation_func_tanh(z):
        return np.tanh(z)

    @staticmethod
    def activation_func_tanh_derivative(a):
        return 1 - np.square(MLP.activation_func_tanh(a))

    @staticmethod
    def activation_func_relu(z):
        return np.maximum(z, 0)

    @staticmethod
    def activation_func_relu_derivative(z):
        return np.array(list(map(lambda x: 0 if x <= 0 else 1, z)))

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

    def normalize_validation_data(self, validation_data, validation_labels):
        res_validation_data = []
        res_validation_outputs = []

        for matrix, output in zip(validation_data, validation_labels):
            res_validation_data.append(matrix.flatten() / 255)
            expected_labels_one_in_n = np.zeros(10)  # output labels count = 10
            expected_labels_one_in_n[output] = 1
            res_validation_outputs.append(expected_labels_one_in_n)

        return zip(res_validation_data, res_validation_outputs)

    def no_optimizer(self, layer, weight_gradients, bias_gradients, batch_size):
        layer.weights = layer.weights - (self.learning_factor / batch_size) * weight_gradients
        layer.biases = layer.biases - self.learning_factor / batch_size * bias_gradients

    def momentum(self, layer, weight_gradients, bias_gradients, batch_size, momentum):
        weight_gradients *= self.learning_factor / batch_size
        bias_gradients *= self.learning_factor / batch_size

        weight_gradients -= layer.previous_weights_delta * momentum
        bias_gradients -= layer.previous_biases_delta * momentum

        layer.weights -= weight_gradients
        layer.biases -= bias_gradients

        layer.previous_weights_delta = copy.deepcopy(weight_gradients)
        layer.previous_biases_delta = copy.deepcopy(bias_gradients)

    def nesterov_momentum(self, layer, weight_gradients, bias_gradients, batch_size, momentum):
        weight_gradients *= self.learning_factor / batch_size
        bias_gradients *= self.learning_factor / batch_size

        weight_gradients -= layer.previous_weights_delta * momentum
        bias_gradients -= layer.previous_biases_delta * momentum

        layer.weights -= weight_gradients + weight_gradients * momentum
        layer.biases -= bias_gradients + bias_gradients * momentum

        layer.previous_weights_delta = copy.deepcopy(weight_gradients)
        layer.previous_biases_delta = copy.deepcopy(bias_gradients)

    def adagrad(self, layer, weight_gradients, bias_gradients, batch_size):
        epsilon = 1e-8
        layer.weight_accumulators += weight_gradients ** 2
        layer.bias_accumulators += bias_gradients ** 2

        layer.weights -= self.learning_factor / batch_size * weight_gradients / (
                np.sqrt(layer.weight_accumulators) + epsilon)
        layer.biases -= self.learning_factor / batch_size + bias_gradients / (
                np.sqrt(layer.bias_accumulators) + epsilon)

    def adadelta(self, layer, weight_gradients, bias_gradients, batch_size):
        decay_rate = 0.999
        epsilon = 1e-8

        layer.weight_accumulators = decay_rate * layer.weight_accumulators + (1 - decay_rate) * weight_gradients ** 2
        layer.bias_accumulators = decay_rate * layer.bias_accumulators + (1 - decay_rate) * bias_gradients ** 2

        layer.weights -= self.learning_factor / batch_size * weight_gradients / (
                np.sqrt(layer.weight_accumulators) + epsilon)
        layer.biases -= self.learning_factor / batch_size * bias_gradients / (
                np.sqrt(layer.bias_accumulators) + epsilon)

    def adam(self, layer, weight_gradients, bias_gradients, batch_size):
        epsilon = 1e-8
        beta1 = 0.9
        beta2 = 0.999

        layer.weight_accumulators = beta1 * layer.weight_accumulators + (1 - beta1) * weight_gradients
        layer.v_weight_accumulators = beta2 * layer.v_weight_accumulators + (1 - beta2) * (weight_gradients ** 2)
        layer.bias_accumulators = beta1 * layer.bias_accumulators + (1 - beta1) * bias_gradients
        layer.v_bias_accumulators = beta2 * layer.v_bias_accumulators + (1 - beta2) * (bias_gradients ** 2)

        layer.weights -= self.learning_factor / batch_size * layer.weight_accumulators / (
                np.sqrt(layer.v_weight_accumulators) + epsilon)
        layer.weights -= self.learning_factor / batch_size * layer.bias_accumulators / (
                np.sqrt(layer.v_bias_accumulators) + epsilon)
