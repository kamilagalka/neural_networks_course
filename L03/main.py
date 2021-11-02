import logging

import idx2numpy
import matplotlib.pyplot as plt  # noqa

from mlp_batch import MLP, Layer, init_weights

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
    TRAINING_DATA = read_data(data_file_names["train_data"])
    TRAINING_LABELS = read_data(data_file_names["train_labels"])

    image_input_vector_size = len(TRAINING_DATA[0].flatten())
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

    mlp = MLP(
        layers=layers,
        learning_factor=1,
    )
    pixels = TRAINING_DATA[55000]
    plt.imshow(pixels, cmap='gray')
    plt.show()

    pixels = TRAINING_DATA[55040]
    plt.imshow(pixels, cmap='gray')
    plt.show()

    pixels = TRAINING_DATA[56000]
    plt.imshow(pixels, cmap='gray')
    plt.show()

    mlp.train(TRAINING_DATA[:50000], TRAINING_LABELS[:50000])


    prediction = mlp.predict(TRAINING_DATA[55000])
    logging.info(f"{prediction} <-> {TRAINING_LABELS[55000]}")

    prediction = mlp.predict(TRAINING_DATA[55040])
    logging.info(f"{prediction} <-> {TRAINING_LABELS[55040]}")

    prediction = mlp.predict(TRAINING_DATA[56000])
    logging.info(f"{prediction} <-> {TRAINING_LABELS[56000]}")
