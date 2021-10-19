import logging

import idx2numpy
import matplotlib.pyplot as plt  # noqa

from mlp import MLP

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

    # pixels = TRAINING_DATA[0]
    # plt.imshow(pixels, cmap='gray')
    # plt.show()

    image_input_vector_size = len(TRAINING_DATA[0].flatten())
    logging.info(image_input_vector_size)

    mlp = MLP(
        starting_neurons_count=image_input_vector_size,
        layers_count=2,
        output_labels_count=10,
        learning_factor=None
    )

    predicted_labels = mlp.train(TRAINING_DATA, TRAINING_LABELS)
    logging.info(predicted_labels)
    logging.info(len(predicted_labels))
    logging.info(sum(predicted_labels))
