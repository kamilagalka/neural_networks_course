import logging
import random

import numpy as np

from L01.perceptron import PerceptronBipolar
from L02.adaline import Adaline

TRAIN_INPUT_SET_SIZE = 1

ALPHA = 0.1

MI = 0.15
EPSILON = 0.6

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

if __name__ == '__main__':
    for _ in range(10):
        START_WEIGHTS = np.array([random.uniform(-0.1, 0.1) for _ in range(3)])

        start_weights = np.copy(START_WEIGHTS)

        p = PerceptronBipolar(alpha=ALPHA, weights=start_weights)
        p.train(TRAIN_INPUT, TRAIN_OUTPUT)
        logging.info(f"Perceptron training finished in {p.iteration_counter} iterations")

        start_weights = np.copy(START_WEIGHTS)

        a = Adaline(MI, start_weights, EPSILON)
        a.train(TRAIN_INPUT, TRAIN_OUTPUT)
