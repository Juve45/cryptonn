

# Importing necessary libraries
import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
import cifar_loader as cl
import numpy as np


IMG_COLOR = 3
IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_SZ = IMG_HEIGHT * IMG_WIDTH 
LAYER_1 = 700
LABEL_CNT = 10

x_init, y = cl.load_data('cifar10/data_batch_1')

y = y[:70]
print(y)

def grayscale(pixels):

    ans = []
    for i in range(IMG_SZ):
        ans.append(int(0.299*pixels[i] + 0.587*pixels[i + 1024] + 0.114*pixels[i + 2048]))
    # print(ans)
    return np.array(ans)


x = [grayscale(i).reshape(32, 32, 1) for i in x_init[:5]]

# print("x=", x[0].tolist())

print(x[0].shape)


# # Peforming reshaping operation
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# # Normalization
# x_train = x_train / 255
# x_test = x_test / 255

# # One Hot Encoding
# y_train = keras.utils.to_categorical(y_train, 10)
# y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()
# Select 6 feature convolution kernels with a size of 5 * 5 (without offset), and get 66 feature maps. The size of each feature map is 32−5 + 1 = 2832−5 + 1 = 28.
# That is, the number of neurons has been reduced from 10241024 to 28 ∗ 28 = 784 28 ∗ 28 = 784.
# Parameters between input layer and C1 layer: 6 ∗ (5 ∗ 5 + 1)
model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
# The input matrix size of this layer is 14 * 14 * 6, the filter size used is 5 * 5, and the depth is 16. This layer does not use all 0 padding, and the step size is 1.
# The output matrix size of this layer is 10 * 10 * 16. This layer has 5 * 5 * 6 * 16 + 16 = 2416 parameters
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
# The input matrix size of this layer is 10 * 10 * 16. The size of the filter used in this layer is 2 * 2, and the length and width steps are both 2, so the output matrix size of this layer is 5 * 5 * 16.
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model.summary()
model.save('model.h5')
# model.fit([], [], batch_size=128, epochs=20, verbose=1, validation_data=([], []))


from keras import backend as K

inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions

# Testing
# test = np.random.random(input_shape)[np.newaxis,...]
test = x[0]
layer_outs = [func([test, 1.]) for func in functors]
print(layer_outs)
