import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense,Dropout,Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# load the dataset

(x_train, y_train), (x_test, y_test)=mnist.load_data()

# compute the lenght of labels

num_labels=len(np.unique(y_train))

# convert to one-hot vector

y_train = to_categorical(y_train)
y_test= to_categorical(y_test)

# the image dimension

image_size= x_train.shape[1]

# resize and normalize

x_train = np.reshape(x_train, [-1 , image_size, image_size, 1])
x_test=np.reshape(x_test, [-1, image_size, image_size, 1])
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255


# network parameters
input_shape=(image_size, image_size, 1)
batch_size=128
dropout=0.3
kernel_size=3
pool_size=2
filters=64

# use functionnal API to build cn layers

inputs=Input(shape=input_shape)
y=Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(inputs)
y=MaxPooling2D()(y)
y=Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(y)
y=MaxPooling2D()(y)
y=Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(y)
# image to vector before connecting to dense layer
y=Flatten()(y)
# dropout regularization
y=Dropout(dropout)(y)
outputs=Dense(num_labels, activation='softmax')(y)

# build the model by supplying inputs/outputs
model=Model(inputs=inputs, outputs=outputs)
# network model in text
model.summary()


# classifier loss, Adam optimizer, classifier accuracy

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# train the model with input images and labels

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=batch_size)

# model accuracy on test dataset

score= model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)

print("\n Test accuracy : %.1f%%" % (100.0*score[1]))