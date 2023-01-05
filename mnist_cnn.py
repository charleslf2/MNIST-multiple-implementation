# import the package

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# load the datasets

(x_train, y_train),(x_test, y_test)= mnist.load_data()

# compute the number of labels

num_labels=len(np.unique(y_train))

# convert to one-hot vector

y_train= to_categorical(y_train)
y_test=to_categorical(y_test)

# input image dimensions

image_size=x_train.shape[1]

# resize and normalize

x_train=np.reshape(x_train, [-1, image_size, image_size,1])
x_test=np.reshape(x_test, [-1, image_size, image_size, 1])
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255

# netwok parameters

# image is processed as is (square grayscale)

input_shape=(image_size, image_size, 1)
batch_size=128
kernel_size=3
pool_size=2
filters=64
dropout=0.2

# model is a stack of CNN-RelU_MaxPooling

model=Sequential()

model.add(Conv2D(filters=filters, kernel_size=kernel_size,
                activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation='relu'))
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation='relu'))
model.add(Flatten())
# droput added as regularizer
model.add(Dropout(dropout))
# output layer is 10-dim one-hot vector
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()

# loss function for one-hot vector
# use of adam optimizer
# accuracy is good metric for classification taks

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# train the network

model.fit(x_train, y_train, epochs=10, batch_size=batch_size)

_, acc= model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)

print("\n Test accuracy : %1f%%" % (100.0 * acc))

                