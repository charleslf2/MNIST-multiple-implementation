import numpy as np 
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.datasets import mnist


# load the dataset

(x_train, y_train),(x_test, y_test) = mnist.load_data()


# compute the number of labels
num_labels=len(np.unique(y_train))

# conver to one-hot vector

y_train= to_categorical(y_train)
y_test= to_categorical(y_test)

# image dimensions 

image_size= x_train.shape[1]
input_size=image_size*image_size

# resize and normalize

x_train= np.reshape(x_train, [-1, input_size])
x_train=x_train.astype('float32')/255
x_test=np.reshape(x_test, [-1, input_size])
x_test=x_test.astype('float32')/255



# network parameters

batch_size=128
hidden_units=256
dropout=0.45

# model is a 3-layer MLP with RelU and dropout after each layer

model=Sequential()
model.add(Dense(hidden_units, input_dim=input_size))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(hidden_units))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(num_labels))
# this is the output for one-hot vector
model.add(Activation('softmax'))
model.summary()
#plot_model(model, to_file='mlp_mnist.png', show_shapes=True)

# loss function for one-hot vector
# use od adma optimizer 
# accuracy is good metric for classification tasks

model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
                )
# train the network
model.fit(x_train, y_train, epochs=20, batch_size=batch_size)

# validate the model on test dataset to determie generalization
_, acc=model.evaluate(x_test, y_test,   
                        batch_size=batch_size,verbose=0)

print("\n Test accuracy : %.1f%%" % (100.0 * acc))