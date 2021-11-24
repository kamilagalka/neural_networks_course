import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
print(f"X_train.shape: {X_train.shape}")

X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
print(f"X_test.shape: {X_test.shape}")

X_train = X_train / 255
X_test = X_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
output_labels_count = y_test.shape[1]

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(output_labels_count, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# plt.imshow(x_train[0])
# plt.show()
# x_train = x_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
# x_train = x_train.astype('float32')
# x_train /= 255
#
# # predict
# out = model.predict(x_train[0:1])
# print(np.argmax(out, axis=1))
