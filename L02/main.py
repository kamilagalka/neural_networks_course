import logging

from adaline import Adaline
import numpy as np
import random

logging.basicConfig(level=logging.INFO)

TRAIN_INPUT_SET_SIZE = 1
MI = 0.1
EPSILON = 0.4

TRAIN_INPUT = []
TRAIN_OUTPUT = []

for _ in range(TRAIN_INPUT_SET_SIZE):
    TRAIN_INPUT.append(np.array([random.uniform(-1.1, -0.9), random.uniform(-1.1, -0.9)]))
    TRAIN_OUTPUT.append(-1)
    TRAIN_INPUT.append(np.array([random.uniform(-1.1, -0.9), random.uniform(0.9, 1.1)]))
    TRAIN_OUTPUT.append(-1)
    TRAIN_INPUT.append(np.array([random.uniform(0.9, 1.1), random.uniform(0.9, 1.1)]))
    TRAIN_OUTPUT.append(1)
    TRAIN_INPUT.append(np.array([random.uniform(0.9, 1.1), random.uniform(-1.1, -0.9)]))
    TRAIN_OUTPUT.append(-1)


START_WEIGHTS = np.array([random.uniform(-0.1, 0.1) for _ in range(3)])

if __name__ == '__main__':
    a = Adaline(mi=MI, weights=START_WEIGHTS, epsilon=EPSILON)

    a.train(TRAIN_INPUT, TRAIN_OUTPUT)

    logging.info("Testing adaline")
    logging.info(f"[-1, -1] -> {a.predict(np.array([-1, -1]))}")
    logging.info(f"[1, 1] -> {a.predict(np.array([1, 1]))}")
    logging.info(f"[-1, 1] -> {a.predict(np.array([-1, 1]))}")
    logging.info(f"[1, -1] -> {a.predict(np.array([1, -1]))}")

    logging.info(f"[-1.0123, -1.0456] -> {a.predict(np.array([-1.0123, -1.0456]))}")
    logging.info(f"[1.0123, -1.0456] -> {a.predict(np.array([1.0123, -1.0456]))}")
    logging.info(f"[-1.0123, 1.0456] -> {a.predict(np.array([-1.0123, 1.0456]))}")
    logging.info(f"[1.0123, 1.0456] -> {a.predict(np.array([1.0123, 1.0456]))}")
