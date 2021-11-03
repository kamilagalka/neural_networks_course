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

if __name__ == "__main__":
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

    layers = [
        Layer(MLP.activation_func_sigm, MLP.activation_func_sigm_derivative,
              init_weights(loc, scale, (image_input_vector_size, 15)), init_weights(loc, scale, (15,))),
        Layer(MLP.activation_func_sigm, MLP.activation_func_sigm_derivative,
              init_weights(loc, scale, (15, 20)), init_weights(loc, scale, (20,))),
        Layer(MLP.softmax, MLP.softmax,
              init_weights(loc, scale, (20, output_size)), init_weights(loc, scale, (10,))),
    ]
    #
    # layers = [
    #     Layer(MLP.activation_func_sigm, MLP.activation_func_sigm_derivative,
    #           np.loadtxt('weights/weights_0.csv', delimiter=','), np.loadtxt('weights/biases_0.csv', delimiter=',')),
    #     Layer(MLP.activation_func_sigm, MLP.activation_func_sigm_derivative,
    #           np.loadtxt('weights/weights_1.csv', delimiter=','), np.loadtxt('weights/biases_1.csv', delimiter=',')),
    #     Layer(MLP.softmax, MLP.softmax,
    #           np.loadtxt('weights/weights_2.csv', delimiter=','), np.loadtxt('weights/biases_2.csv', delimiter=',')),
    # ]

    mlp = MLP(
        layers=layers,
        learning_factor=1,
    )
    mlp.train(TRAINING_DATA, TRAINING_LABELS, VALIDATION_DATA, VALIDATION_LABELS)

    pixels = INPUT_DATA[55000]
    plt.imshow(pixels, cmap='gray')
    plt.show()

    for i in [55000, 56000, 55040, 59999, 54322, 55555, 51234]:
        prediction = mlp.predict(INPUT_DATA[i])
        logging.info(f"{prediction} <-> {INPUT_LABELS[i]}")
        pixels = INPUT_DATA[i]
        plt.imshow(pixels, cmap='gray')
        plt.show()
