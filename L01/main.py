import logging

from perceptron import Perceptron
import numpy as np
import random

logging.basicConfig(level=logging.INFO)

TRAIN_INPUT_SET_SIZE = 1
ALPHA = 0.1

TRAIN_INPUT = []
TRAIN_OUTPUT = []

for _ in range(TRAIN_INPUT_SET_SIZE):
    TRAIN_INPUT.append(np.array([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)]))
    TRAIN_OUTPUT.append(0)
    TRAIN_INPUT.append(np.array([random.uniform(-0.1, 0.1), random.uniform(0.9, 1.1)]))
    TRAIN_OUTPUT.append(0)
    TRAIN_INPUT.append(np.array([random.uniform(0.9, 1.1), random.uniform(-0.1, 0.1)]))
    TRAIN_OUTPUT.append(0)
    TRAIN_INPUT.append(np.array([random.uniform(0.9, 1.1), random.uniform(0.9, 1.1)]))
    TRAIN_OUTPUT.append(1)

START_WEIGHTS = np.array([random.uniform(0, 0.1) for _ in range(3)])

if __name__ == '__main__':
    p = Perceptron(alpha=ALPHA, weights=START_WEIGHTS)

    p.train(TRAIN_INPUT, TRAIN_OUTPUT)

    logging.info("Testing perceptron")
    logging.info(f"[0, 0] -> {p.predict(np.array([0, 0]))}")
    logging.info(f"[0, 1] -> {p.predict(np.array([0, 1]))}")
    logging.info(f"[1, 0] -> {p.predict(np.array([1, 0]))}")
    logging.info(f"[1, 1] -> {p.predict(np.array([1, 1]))}")

    logging.info(f"[0.0123, -0.0456] -> {p.predict(np.array([0.0123, -0.0456]))}")
    logging.info(f"[1.0123, -0.0456] -> {p.predict(np.array([1.0123, -0.0456]))}")
    logging.info(f"[0.0123, 1.0456] -> {p.predict(np.array([0.0123, 1.0456]))}")
    logging.info(f"[1.0123, 1.0456] -> {p.predict(np.array([1.0123, 1.0456]))}")
