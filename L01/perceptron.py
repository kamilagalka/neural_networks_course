import logging
import random

import numpy as np

logging.basicConfig(level=logging.INFO)

# train_input = [
#     np.array([0, 0]),
#     np.array([1, 0]),
#     np.array([0, 1]),
#     np.array([1, 1]),
# ]
#
# train_output = np.array([0, 0, 0, 1])
#
# start_weights = np.array([random.uniform(0, 0.1) for _ in range(3)])

TRAIN_INPUT_SET_SIZE = 1
train_input = []
train_output = []

for _ in range(TRAIN_INPUT_SET_SIZE):
    train_input.append(np.array([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)]))
    train_output.append(0)
    train_input.append(np.array([random.uniform(-0.1, 0.1), random.uniform(0.9, 1.1)]))
    train_output.append(0)
    train_input.append(np.array([random.uniform(0.9, 1.1), random.uniform(-0.1, 0.1)]))
    train_output.append(0)
    train_input.append(np.array([random.uniform(0.9, 1.1), random.uniform(0.9, 1.1)]))
    train_output.append(1)

start_weights = np.array([random.uniform(0, 0.1) for _ in range(3)])


ALPHA = 0.1


def activation_func(x, weights):
    return np.dot(x, weights[:-1]) + weights[-1]


def get_perceptron_output(z):
    return 1 if z > 0 else 0


def get_error(expected, actual):
    return expected - actual


def adjust_weigths(alpha, weights, error, input_data):
    weights[:-1] += alpha * error * input_data
    weights[-1] += alpha * error

    return weights


if __name__ == '__main__':
    # Usage assumption: bias weight as the last weight
    no_errors = False
    current_weights = start_weights

    iteration_counter = 0
    while not no_errors:
        logging.info(f"Iteration no {iteration_counter}")
        iteration_counter += 1
        no_errors = True

        for x, y in zip(train_input, train_output):
            z = activation_func(x, current_weights)
            perceptron_output = get_perceptron_output(z)
            perceptron_error = get_error(expected=y, actual=perceptron_output)
            current_weights = adjust_weigths(alpha=ALPHA, weights=current_weights, error=perceptron_error, input_data=x)
            if perceptron_error != 0:
                no_errors = False

            logging.info(f"{y} <-> {perceptron_output}")

        if no_errors:
            logging.info(f"bias: {current_weights[-1]}")
            logging.info(f"weights: {current_weights[:-1]}")
