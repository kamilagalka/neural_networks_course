import logging
import copy
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

    TRAINING_DATA = INPUT_DATA[:2000]
    TRAINING_LABELS = INPUT_LABELS[:2000]

    VALIDATION_DATA = INPUT_DATA[5000:7000]
    VALIDATION_LABELS = INPUT_LABELS[5000:7000]

    # TEST_DATA = INPUT_DATA[50000:]
    # TEST_LABELS = INPUT_LABELS[50000:]

    image_input_vector_size = len(INPUT_DATA[0].flatten())
    output_size = 10
    loc = 0
    scale = 0.1

    layers = [
        Layer(MLP.activation_func_sigm, MLP.activation_func_sigm_derivative,
              init_weights(scale, (image_input_vector_size, 784), ""), init_weights(scale, (784,), "")),
        Layer(MLP.activation_func_sigm, MLP.activation_func_sigm_derivative,
              init_weights(scale, (784, 100), ""), init_weights(scale, (100,), "")),
        Layer(MLP.activation_func_sigm, MLP.activation_func_sigm_derivative,
              init_weights(scale, (100, 50), ""), init_weights(scale, (50,), "")),
        Layer(MLP.softmax, MLP.softmax,
              init_weights(scale, (50, output_size), ""), init_weights(scale, (output_size, ), "")),
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

    logging.info("============STANDARD==================")

    mlp = MLP(
        layers=copy.deepcopy(layers),
        learning_factor=0.5,
    )
    mlp.train(TRAINING_DATA, TRAINING_LABELS, VALIDATION_DATA, VALIDATION_LABELS)
    
    logging.info("============XAVIER==================")
    
    layers = [
        Layer(MLP.activation_func_sigm, MLP.activation_func_sigm_derivative,
              init_weights(scale, (image_input_vector_size, 784), "Xavier"), init_weights(scale, (784,), "Xavier")),
        Layer(MLP.activation_func_sigm, MLP.activation_func_sigm_derivative,
              init_weights(scale, (784, 100), "Xavier"), init_weights(scale, (100,), "Xavier")),
        Layer(MLP.softmax, MLP.softmax,
              init_weights(scale, (100, output_size), "Xavier"), init_weights(scale, (output_size, ), "Xavier")),
    ]

    mlp = MLP(
        layers=copy.deepcopy(layers),
        learning_factor=0.5,
    )
    mlp.train(TRAINING_DATA, TRAINING_LABELS, VALIDATION_DATA, VALIDATION_LABELS)
    
    logging.info("============HE==================")
    
    layers = [
        Layer(MLP.activation_func_sigm, MLP.activation_func_sigm_derivative,
              init_weights(scale, (image_input_vector_size, 784), "He"), init_weights(scale, (784,), "He")),
        Layer(MLP.activation_func_sigm, MLP.activation_func_sigm_derivative,
              init_weights(scale, (784, 100), "He"), init_weights(scale, (100,), "He")),
        Layer(MLP.softmax, MLP.softmax,
              init_weights(scale, (100, output_size), "He"), init_weights(scale, (output_size, ), "He")),
    ]

    mlp = MLP(
        layers=copy.deepcopy(layers),
        learning_factor=0.5,
    )
    mlp.train(TRAINING_DATA, TRAINING_LABELS, VALIDATION_DATA, VALIDATION_LABELS)

    # logging.info("============MOMENTUM==================")
    #
    #
    # mlp = MLP(
    #     layers=copy.deepcopy(layers),
    #     learning_factor=0.1,
    # )
    # mlp.train(TRAINING_DATA, TRAINING_LABELS, VALIDATION_DATA, VALIDATION_LABELS, optimizer='momentum')
    #
    # logging.info("============NESTEROV==================")
    #
    #
    # mlp = MLP(
    #     layers=copy.deepcopy(layers),
    #     learning_factor=0.1,
    # )
    # mlp.train(TRAINING_DATA, TRAINING_LABELS, VALIDATION_DATA, VALIDATION_LABELS, optimizer='nesterov')
    #
    # logging.info("============ADAGRAD==================")
    #
    # mlp = MLP(
    #     layers=copy.deepcopy(layers),
    #     learning_factor=0.01,
    # )
    # mlp.train(TRAINING_DATA, TRAINING_LABELS, VALIDATION_DATA, VALIDATION_LABELS, optimizer='adagrad')
    #
    #
    # logging.info("============ADADELTA==================")
    #
    # mlp = MLP(
    #     layers=copy.deepcopy(layers),
    #     learning_factor=0.01,
    # )
    # mlp.train(TRAINING_DATA, TRAINING_LABELS, VALIDATION_DATA, VALIDATION_LABELS, optimizer='adadelta')
    #
    # logging.info("============ADAM==================")
    #
    # mlp = MLP(
    #     layers=copy.deepcopy(layers),
    #     learning_factor=0.01,
    # )
    # mlp.train(TRAINING_DATA, TRAINING_LABELS, VALIDATION_DATA, VALIDATION_LABELS, optimizer='adam')
    #

    # for i in [55000, 56000, 55040, 59999, 54322, 55555, 51234]:
    #     prediction = mlp.predict(INPUT_DATA[i])
    #     logging.info(f"{prediction} <-> {INPUT_LABELS[i]}")
    #     pixels = INPUT_DATA[i]
    #     plt.imshow(pixels, cmap='gray')
    #     plt.show()
