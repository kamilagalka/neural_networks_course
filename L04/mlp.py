from keras.models import Sequential
from keras.layers import Dense

output_labels_count = 10
num_pixels = 784


def mlp():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(output_labels_count, kernel_initializer='normal', activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

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
