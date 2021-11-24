from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_pixels = X_train.shape[1] * X_train.shape[2]

X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
# print(f"X_train.shape: {X_train.shape}")

X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')
# print(f"X_test.shape: {X_test.shape}")

X_train = X_train / 255
X_test = X_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
output_labels_count = y_test.shape[1]

model = Sequential()
model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
model.add(Dense(100, kernel_initializer='normal', activation='relu'))
model.add(Dense(output_labels_count, kernel_initializer='normal', activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)
print("MLP Error: %.2f%%" % (100 - scores[1] * 100))

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# plt.imshow(x_train[0])
# plt.show()
# x_train = x_train.reshape(60000, 784)
# x_train = x_train.astype('float32')
# x_train /= 255
#
# # predict
# out = model.predict(x_train[0:1])
# print(np.argmax(out, axis=1))
