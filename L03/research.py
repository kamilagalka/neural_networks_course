import logging

import idx2numpy
import matplotlib.pyplot as plt  # noqa

from mlp_batch import MLP, Layer, init_weights
import numpy as np

logging.basicConfig(level=logging.INFO)


def read_data(file_name):
    data = idx2numpy.convert_from_file(file_name)
    return data


data_file_names = {
    "train_data": "data/train-images.idx3-ubyte",
    "train_labels": "data/train-labels.idx1-ubyte",

    "test_images": "data/t10k-images.idx3-ubyte",
    "test_labels": "data/t10k-labels.idx1-ubyte"
}

INPUT_DATA = read_data(data_file_names["train_data"])
INPUT_LABELS = read_data(data_file_names["train_labels"])

TRAINING_DATA = INPUT_DATA[:40000]
TRAINING_LABELS = INPUT_LABELS[:40000]

VALIDATION_DATA = INPUT_DATA[40000:50000]
VALIDATION_LABELS = INPUT_LABELS[40000:50000]

TEST_DATA = INPUT_DATA[50000:]
TEST_LABELS = INPUT_LABELS[50000:]

image_input_vector_size = len(INPUT_DATA[0].flatten())
output_size = 10
loc = 0
scale = 0.1


def task_a():
    # a. szybkość uczenia (w epokach) i skuteczność w przypadku różnej liczby neuronów w warstwie
    # ukrytej,
    logging.info("================= TASK A =================")

    for neurons_in_hidden_layer in [10, 15, 100, 300, 784]:
        logging.info(f"-------------- Current neurons in hidden layer: {neurons_in_hidden_layer}")
        layers = [
            Layer(MLP.activation_func_sigm, MLP.activation_func_sigm_derivative,
                  init_weights(loc, scale, (image_input_vector_size, neurons_in_hidden_layer)),
                  init_weights(loc, scale, (neurons_in_hidden_layer,))),
            Layer(MLP.softmax, MLP.softmax,
                  init_weights(loc, scale, (neurons_in_hidden_layer, output_size)), init_weights(loc, scale, (10,))),
        ]

        mlp = MLP(
            layers=layers,
            learning_factor=0.1,
        )
        mlp.train(TRAINING_DATA, TRAINING_LABELS, VALIDATION_DATA, VALIDATION_LABELS)


def task_b():
    # b. wpływ różnych wartości współczynniki uczenia,
    logging.info("================= TASK B =================")
    layers = [
        Layer(MLP.activation_func_sigm, MLP.activation_func_sigm_derivative,
              init_weights(loc, scale, (image_input_vector_size, 15)), init_weights(loc, scale, (15,))),
        Layer(MLP.activation_func_sigm, MLP.activation_func_sigm_derivative,
              init_weights(loc, scale, (15, 20)), init_weights(loc, scale, (20,))),
        Layer(MLP.softmax, MLP.softmax,
              init_weights(loc, scale, (20, output_size)), init_weights(loc, scale, (10,))),
    ]

    for learning_factor in [0.01, 0.1, 0.5, 1, 10]:
        logging.info(f"-------------- Current learning factor: {learning_factor}")
        mlp = MLP(
            layers=layers.copy(),
            learning_factor=learning_factor,
        )
        mlp.train(TRAINING_DATA, TRAINING_LABELS, VALIDATION_DATA, VALIDATION_LABELS)


def task_c():
    # c. wpływ wielkości paczki (batcha),
    logging.info("================= TASK C =================")

    layers = [
        Layer(MLP.activation_func_sigm, MLP.activation_func_sigm_derivative,
              init_weights(loc, scale, (image_input_vector_size, 15)), init_weights(loc, scale, (15,))),
        Layer(MLP.activation_func_sigm, MLP.activation_func_sigm_derivative,
              init_weights(loc, scale, (15, 20)), init_weights(loc, scale, (20,))),
        Layer(MLP.softmax, MLP.softmax,
              init_weights(loc, scale, (20, output_size)), init_weights(loc, scale, (10,))),
    ]

    for bs in [1, 5, 10, 50, 100, 1000, 20000]:
        logging.info(f"-------------- Current batch size: {bs}")
        mlp = MLP(
            layers=layers.copy(),
            learning_factor=0.1,
        )
        mlp.train(TRAINING_DATA, TRAINING_LABELS, VALIDATION_DATA, VALIDATION_LABELS, batch_size=bs)


def task_d():
    # d. wpływ inicjalizacji wartości wag początkowych
    logging.info("================= TASK D =================")

    for loc in [-1, 0, 1]:
        for scale in [0.1, 0.5, 1, 10]:
            logging.info(f"-------------- Current loc: {loc}, current scale: {scale}")
            layers = [
                Layer(MLP.activation_func_sigm, MLP.activation_func_sigm_derivative,
                      init_weights(loc, scale, (image_input_vector_size, 15)), init_weights(loc, scale, (15,))),
                Layer(MLP.activation_func_sigm, MLP.activation_func_sigm_derivative,
                      init_weights(loc, scale, (15, 20)), init_weights(loc, scale, (20,))),
                Layer(MLP.softmax, MLP.softmax,
                      init_weights(loc, scale, (20, output_size)), init_weights(loc, scale, (10,))),
            ]

            mlp = MLP(
                layers=layers,
                learning_factor=0.1,
            )
            mlp.train(TRAINING_DATA, TRAINING_LABELS, VALIDATION_DATA, VALIDATION_LABELS)


def task_e():
    # e. wpływ funkcji aktywacji (tanh lub ReLU)
    logging.info("================= TASK E =================")

    for activation_func, activation_func_der in zip(
            [MLP.activation_func_sigm, MLP.activation_func_tanh, MLP.activation_func_relu],
            [MLP.activation_func_sigm_derivative, MLP.activation_func_tanh_derivative,
             MLP.activation_func_relu_derivative]):
        logging.info(f"--------------")
        layers = [
            Layer(activation_func, activation_func_der,
                  init_weights(loc, scale, (image_input_vector_size, 15)),
                  init_weights(loc, scale, (15,))),
            Layer(MLP.softmax, MLP.softmax,
                  init_weights(loc, scale, (15, output_size)),
                  init_weights(loc, scale, (10,))),
        ]

        mlp = MLP(
            layers=layers,
            learning_factor=0.1,
        )
        mlp.train(TRAINING_DATA, TRAINING_LABELS, VALIDATION_DATA, VALIDATION_LABELS)


if __name__ == '__main__':
    task_a()
    task_b()
    task_c()
    task_d()
    task_e()
