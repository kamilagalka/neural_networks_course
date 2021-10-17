import logging

from adaline import Adaline
import numpy as np
import random

logging.basicConfig(level=logging.INFO)

TRAIN_INPUT_SET_SIZE = 4
MI = 0.2
EPSILON = 0.4

TRAIN_INPUT = []
TRAIN_OUTPUT = []

for _ in range(TRAIN_INPUT_SET_SIZE):
    TRAIN_INPUT.append(np.array([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)]))
    TRAIN_OUTPUT.append(-1)
    TRAIN_INPUT.append(np.array([random.uniform(-0.1, 0.1), random.uniform(0.9, 1.1)]))
    TRAIN_OUTPUT.append(-1)
    TRAIN_INPUT.append(np.array([random.uniform(0.9, 1.1), random.uniform(-0.1, 0.1)]))
    TRAIN_OUTPUT.append(-1)
    TRAIN_INPUT.append(np.array([random.uniform(0.9, 1.1), random.uniform(0.9, 1.1)]))
    TRAIN_OUTPUT.append(1)

START_WEIGHTS = np.array([random.uniform(-0.1, 0.1) for _ in range(3)])


def task_1():
    logging.info("---------------TASK 1---------------")
    range_min = -1
    range_max = 1
    range_step = 0.1

    while range_min < -0.1:
        start_weights = np.array([random.uniform(range_min, range_max) for _ in range(3)])

        a = Adaline(MI, start_weights, EPSILON)
        a.train(TRAIN_INPUT, TRAIN_OUTPUT)

        logging.info(
            f"Training finished in {a.iteration_counter} iterations for start weights in range: {round(range_min, 2)} - {round(range_max, 2)}")

        range_min += range_step
        range_max -= range_step

    logging.info("-----------------------------")

    range_min = -1
    range_max = -0.9

    while range_min < 2:
        start_weights = np.array([random.uniform(range_min, range_max) for _ in range(3)])

        a = Adaline(MI, start_weights, EPSILON)
        a.train(TRAIN_INPUT, TRAIN_OUTPUT)

        logging.info(
            f"Training finished in {a.iteration_counter} iterations for start weights in range: {round(range_min, 2)} - {round(range_max, 2)}")

        range_min += range_step
        range_max += range_step


def task_2():
    logging.info("---------------TASK 2---------------")

    mi = 0.0001

    while mi <= 0.5:
        start_weights = np.copy(START_WEIGHTS)

        a = Adaline(mi, start_weights, EPSILON)
        a.train(TRAIN_INPUT, TRAIN_OUTPUT)

        logging.info(f"Training finished in {a.iteration_counter} iterations for mi: {round(mi, 2)}")
        mi += 0.01


def task_3():
    logging.info("---------------TASK 3---------------")

    epsilon = 0.9

    while epsilon > 0:
        start_weights = np.append(np.copy(START_WEIGHTS)[:-1], epsilon)

        a = Adaline(MI, start_weights, epsilon)
        a.train(TRAIN_INPUT, TRAIN_OUTPUT)

        logging.info(f"Training finished in {a.iteration_counter} iterations for epsilon: {round(epsilon, 2)}")
        epsilon -= 0.05


if __name__ == '__main__':
    task_1()
    task_2()
    task_3()
