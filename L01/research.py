import logging

from perceptron import Perceptron, PerceptronBipolar
import numpy as np
import random

logging.basicConfig(level=logging.INFO)

TRAIN_INPUT_SET_SIZE = 1
ALPHA = 0.0001

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

START_WEIGHTS = np.array([random.uniform(-0.1, 0.1) for _ in range(3)])


def task_1():
    logging.info("---------------TASK 1---------------")

    theta = 0.1

    while theta < 1:
        start_weights = np.append(np.copy(START_WEIGHTS)[:-1], theta)

        p = Perceptron(ALPHA, start_weights, use_bias=False)
        p.train(TRAIN_INPUT, TRAIN_OUTPUT)

        logging.info(f"Training finished in {p.iteration_counter} iterations for theta: {theta}")
        theta += 0.05


def task_2():
    logging.info("---------------TASK 2---------------")
    range_min = -1
    range_max = 1
    range_step = 0.1

    while range_min < -0.1:
        start_weights = np.array([random.uniform(range_min, range_max) for _ in range(3)])

        p = Perceptron(ALPHA, start_weights, use_bias=True)
        p.train(TRAIN_INPUT, TRAIN_OUTPUT)

        logging.info(
            f"Training finished in {p.iteration_counter} iterations for start weights in range: {range_min} - {range_max}")

        range_min += range_step
        range_max -= range_step


def task_3():
    logging.info("---------------TASK 3---------------")

    alpha = 0.1

    while alpha >= 0.0001:
        start_weights = np.copy(START_WEIGHTS)

        p = Perceptron(alpha, start_weights, use_bias=True)
        p.train(TRAIN_INPUT, TRAIN_OUTPUT)

        logging.info(f"Training finished in {p.iteration_counter} iterations for alpha: {alpha}")
        alpha -= 0.001


def task_4():
    logging.info("---------------TASK 4---------------")

    start_weights = np.copy(START_WEIGHTS)
    p = Perceptron(alpha=ALPHA, weights=start_weights, use_bias=True)
    p.train(TRAIN_INPUT, TRAIN_OUTPUT)
    logging.info(f"Training unary perceptron finished in {p.iteration_counter} iterations")

    train_output_bipolar = np.copy(TRAIN_OUTPUT)
    train_output_bipolar[train_output_bipolar == 0] = -1
    train_input_bipolar = np.copy(TRAIN_INPUT)
    train_input_bipolar[train_input_bipolar == 0] = -1

    start_weights = np.copy(START_WEIGHTS)
    p = PerceptronBipolar(alpha=ALPHA, weights=start_weights, use_bias=True)
    p.train(TRAIN_INPUT, train_output_bipolar)
    logging.info(f"Training bipolar perceptron finished in {p.iteration_counter} iterations")


if __name__ == '__main__':
    task_1()
    task_2()
    task_3()
    task_4()
