import copy
import time

from keras.utils import np_utils
from keras.datasets import mnist
from keras_preprocessing.image import ImageDataGenerator
import mlp
import cnn

NUM_OF_RETRIES = 10


def load_data_cnn():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train[:2000]
    y_train = y_train[:2000]
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')

    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')

    X_train = X_train / 255
    X_test = X_test / 255

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    return X_train, y_train, X_test, y_test


def load_data_mlp():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train[:2000]
    y_train = y_train[:2000]
    num_pixels = X_train.shape[1] * X_train.shape[2]

    X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
    X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')

    X_train = X_train / 255
    X_test = X_test / 255

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return X_train, y_train, X_test, y_test


def fit_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
    return model


def fit_model_rotation(model, X_train, y_train, X_test, y_test):
    batch_size = 32
    generator = ImageDataGenerator(rotation_range=15)

    generator_train = generator.flow(X_train, y_train, batch_size=batch_size)
    generator_test = generator.flow(X_test, y_test, batch_size=batch_size)

    model.fit(generator_train, validation_data=generator_test, steps_per_epoch=X_train.shape[0] // batch_size,
              epochs=10)
    return model


def fit_model_worsen(model, X_train, y_train, X_test, y_test):
    batch_size = 32
    generator = ImageDataGenerator(rotation_range=180, width_shift_range=0.99, height_shift_range=0.99,
                                   shear_range=0.99)

    generator_train = generator.flow(X_train, y_train, batch_size=batch_size)
    generator_test = generator.flow(X_test, y_test, batch_size=batch_size)

    model.fit(generator_train, validation_data=generator_test, steps_per_epoch=X_train.shape[0] // batch_size,
              epochs=10)
    return model


def fit_model_shift(model, X_train, y_train, X_test, y_test):
    batch_size = 32
    generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1)

    generator_train = generator.flow(X_train, y_train, batch_size=batch_size)
    generator_test = generator.flow(X_test, y_test, batch_size=batch_size)

    model.fit(generator_train, validation_data=generator_test, steps_per_epoch=X_train.shape[0] // batch_size,
              epochs=10)
    return model


def fit_model_shearing(model, X_train, y_train, X_test, y_test):
    batch_size = 32
    generator = ImageDataGenerator(shear_range=0.2)

    generator_train = generator.flow(X_train, y_train, batch_size=batch_size)
    generator_test = generator.flow(X_test, y_test, batch_size=batch_size)

    model.fit(generator_train, validation_data=generator_test, steps_per_epoch=X_train.shape[0] // batch_size,
              epochs=10)
    return model


def evaluate_model(model, X_test, y_test):
    scores = model.evaluate(X_test, y_test, verbose=0)
    return 100 - scores[1] * 100
    # print(f"{model_name}: {100 - scores[1] * 100}")


if __name__ == '__main__':

    print("-----------------TESTING-----------------")
    mlp_errors = []
    mlp_times = []
    simple_cnn_errors = []
    simple_cnn_times = []
    cnn_pooling_errors = []
    cnn_pooling_times = []
    cnn_bn_errors = []
    cnn_bn_times = []
    cnn_do_errors = []
    cnn_do_times = []
    cnn_rotation_errors = []
    cnn_rotation_times = []
    cnn_shift_errors = []
    cnn_shift_times = []
    cnn_shearing_errors = []
    cnn_shearing_times = []
    cnn_worsen_errors = []
    cnn_worsen_times = []
    X_train_mlp, y_train_mlp, X_test_mlp, y_test_mlp = load_data_mlp()
    X_train_cnn, y_train_cnn, X_test_cnn, y_test_cnn = load_data_cnn()

    for try_number in range(NUM_OF_RETRIES):
        print(f"------------------------------- TRY {try_number} -------------------------------")

        # print("-----------------MLP-----------------")
        # mlp_model = mlp.mlp()
        # start = time.time()
        # mlp_model = fit_model(mlp_model, copy.deepcopy(X_train_mlp), copy.deepcopy(y_train_mlp),
        #                       copy.deepcopy(X_test_mlp),
        #                       copy.deepcopy(y_test_mlp))
        # mlp_errors.append(evaluate_model(mlp_model, X_test_mlp, y_test_mlp))
        # end = time.time()
        # mlp_times.append(end - start)
        #
        # print("-----------------SIMPLE CNN-----------------")
        # simple_cnn = cnn.simple_cnn()
        # start = time.time()
        # simple_cnn = fit_model(simple_cnn, copy.deepcopy(X_train_cnn), copy.deepcopy(y_train_cnn),
        #                        copy.deepcopy(X_test_cnn),
        #                        copy.deepcopy(y_test_cnn))
        # simple_cnn_errors.append(evaluate_model(simple_cnn, X_test_cnn, y_test_cnn))
        # end = time.time()
        # simple_cnn_times.append(end - start)
        #
        # print("-----------------CNN WITH POOLING-----------------")
        # cnn_pooling = cnn.cnn_with_pooling()
        # start = time.time()
        # cnn_pooling = fit_model(cnn_pooling, copy.deepcopy(X_train_cnn), copy.deepcopy(y_train_cnn),
        #                         copy.deepcopy(X_test_cnn),
        #                         copy.deepcopy(y_test_cnn))
        # cnn_pooling_errors.append(evaluate_model(cnn_pooling, X_test_cnn, y_test_cnn))
        # end = time.time()
        # cnn_pooling_times.append(end - start)

        # print("-----------------CNN BATCH NORMALIZATION-----------------")
        # cnn_with_bn = cnn.cnn_with_batch_normalization()
        # start = time.time()
        # cnn_with_bn = fit_model(cnn_with_bn, copy.deepcopy(X_train_cnn), copy.deepcopy(y_train_cnn),
        #                         copy.deepcopy(X_test_cnn),
        #                         copy.deepcopy(y_test_cnn))
        # cnn_bn_errors.append(evaluate_model(cnn_with_bn, X_test_cnn, y_test_cnn))
        # end = time.time()
        # cnn_bn_times.append(end - start)

        # print("-----------------CNN DROPOUT-----------------")
        # cnn_with_dropout = cnn.cnn_with_dropout()
        # start = time.time()
        # cnn_with_dropout = fit_model(cnn_with_dropout, copy.deepcopy(X_train_cnn), copy.deepcopy(y_train_cnn),
        #                              copy.deepcopy(X_test_cnn),
        #                              copy.deepcopy(y_test_cnn))
        # cnn_do_errors.append(evaluate_model(cnn_with_dropout, X_test_cnn, y_test_cnn))
        # end = time.time()
        # cnn_do_times.append(end - start)

        # print("-----------------CNN + ROTATION-----------------")
        # cnn_rotation = cnn.cnn_with_dropout()
        # start = time.time()
        # cnn_rotation = fit_model_rotation(cnn_rotation, copy.deepcopy(X_train_cnn), copy.deepcopy(y_train_cnn),
        #                                   copy.deepcopy(X_test_cnn),
        #                                   copy.deepcopy(y_test_cnn))
        # cnn_rotation_errors.append(evaluate_model(cnn_rotation, X_test_cnn, y_test_cnn))
        # end = time.time()
        # cnn_rotation_times.append(end - start)

        # print("-----------------CNN + SHIFTING-----------------")
        # cnn_shift = cnn.cnn_with_dropout()
        # start = time.time()
        # cnn_shift = fit_model_shift(cnn_shift, copy.deepcopy(X_train_cnn), copy.deepcopy(y_train_cnn),
        #                             copy.deepcopy(X_test_cnn),
        #                             copy.deepcopy(y_test_cnn))
        # cnn_shift_errors.append(evaluate_model(cnn_shift, X_test_cnn, y_test_cnn))
        # end = time.time()
        # cnn_shift_times.append(end - start)
        #
        # print("-----------------CNN + SHEARING-----------------")
        # cnn_shearing = cnn.cnn_with_dropout()
        # start = time.time()
        # cnn_shearing = fit_model_shift(cnn_shearing, copy.deepcopy(X_train_cnn), copy.deepcopy(y_train_cnn),
        #                                copy.deepcopy(X_test_cnn),
        #                                copy.deepcopy(y_test_cnn))
        # cnn_shearing_errors.append(evaluate_model(cnn_shearing, X_test_cnn, y_test_cnn))
        # end = time.time()
        # cnn_shearing_times.append(end - start)
        #
        print("-----------------CNN WORSEN-----------------")
        cnn_worsen = cnn.cnn_with_dropout()
        start = time.time()
        cnn_worsen = fit_model_worsen(cnn_worsen, copy.deepcopy(X_train_cnn), copy.deepcopy(y_train_cnn),
                                      copy.deepcopy(X_test_cnn),
                                      copy.deepcopy(y_test_cnn))
        cnn_worsen_errors.append(evaluate_model(cnn_worsen, X_test_cnn, y_test_cnn))
        end = time.time()
        cnn_worsen_times.append(end - start)

        # print(f'MLP error: {mlp_errors[-1]}')
        # print(f'MLP time: {mlp_times[-1]}')
        # print(f'Simple CNN error: {simple_cnn_errors[-1]}')
        # print(f'Simple CNN time: {simple_cnn_times[-1]}')
        # print(f'CNN with pooling error: {cnn_pooling_errors[-1]}')
        # print(f'CNN with pooling time: {cnn_pooling_times[-1]}')
        # print(f'CNN with BN error: {cnn_bn_errors[-1]}')
        # print(f'CNN with BN time: {cnn_bn_times[-1]}')
        # print(f'CNN with Dropout error: {cnn_do_errors[-1]}')
        # print(f'CNN with Dropout time: {cnn_do_times[-1]}')
        # print(f'CNN with rotation error: {cnn_rotation_errors[-1]}')
        # print(f'CNN with rotation time: {cnn_rotation_times[-1]}')
        # print(f'CNN with shift error: {cnn_shift_errors[-1]}')
        # print(f'CNN with shift time: {cnn_shift_times[-1]}')
        # print(f'CNN with shearing error: {cnn_shearing_errors[-1]}')
        # print(f'CNN with shearing time: {cnn_shearing_times[-1]}')
        print(f'CNN worsen error: {cnn_worsen_errors[-1]}')
        print(f'CNN worsen time: {cnn_worsen_times[-1]}')

    print("---------------------------------------------------------------")
    # print(f'MLP error: {sum(mlp_errors) / NUM_OF_RETRIES}')
    # print(f'MLP time: {sum(mlp_times) / NUM_OF_RETRIES}')
    # print(f'Simple CNN error: {sum(simple_cnn_errors) / NUM_OF_RETRIES}')
    # print(f'Simple CNN time: {sum(simple_cnn_times) / NUM_OF_RETRIES}')
    # print(f'CNN with pooling error: {sum(cnn_pooling_errors) / NUM_OF_RETRIES}')
    # print(f'CNN with pooling time: {sum(cnn_pooling_times) / NUM_OF_RETRIES}')
    # print(f'CNN with bn error: {sum(cnn_bn_errors) / NUM_OF_RETRIES}')
    # print(f'CNN with bn time: {sum(cnn_bn_times) / NUM_OF_RETRIES}')
    # print(f'CNN with do error: {sum(cnn_do_errors) / NUM_OF_RETRIES}')
    # print(f'CNN with do time: {sum(cnn_do_times) / NUM_OF_RETRIES}')
    # print(f'CNN with rotation error: {sum(cnn_rotation_errors) / NUM_OF_RETRIES}')
    # print(f'CNN with rotation time: {sum(cnn_rotation_times) / NUM_OF_RETRIES}')
    # print(f'CNN with shifting error: {sum(cnn_shift_errors) / NUM_OF_RETRIES}')
    # print(f'CNN with shifting time: {sum(cnn_shift_times) / NUM_OF_RETRIES}')
    # print(f'CNN with shearing error: {sum(cnn_shearing_errors) / NUM_OF_RETRIES}')
    # print(f'CNN with shearing time: {sum(cnn_shearing_times) / NUM_OF_RETRIES}')
    print(f'CNN with worsen error: {sum(cnn_worsen_errors) / NUM_OF_RETRIES}')
    print(f'CNN with worsen time: {sum(cnn_worsen_times) / NUM_OF_RETRIES}')
