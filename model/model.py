import gzip
import numpy as numpy

# Import keras
import keras as kr
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

import tensorflow as tf

# %matplotlib inline # Only usable in iPython
import matplotlib.pyplot as plot

# For encoding categorical variables
import sklearn.preprocessing as skpre

# The MNIST database contains 60,000 training images and 10,000 test images.
# x_train and x_test contain the greyscale RGB codes (0 - 255) for the training images and test images, respectively.
# y_train and y_test contain the labels (0 - 9) for the training images and test images, respectively.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

image_index = 7777 # You may select anything up to 60,000
print(y_train[image_index]) # Displays the appropriate label for the image at the specified index
plot.imshow(x_train[image_index], cmap='Greys')
# plot.show()

# For the convolutional neural network, we need to know the shape of the dataset to channel it into the network. 
# print(x_train.shape)

# To be able to use the dataset with Keras, the numpy arrays need to be 4-dimensional. However, from above (print(x_train.shape)), 
# you can see that the array is 3-dimensional. The data must be normalised (requirement of neural network models). This can be done by dividing
# the RGB codes (both the training and test images) by 255

# Here, we 'reshape' the the 3D arrays to 4D arrays to work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# float32 is used to ensure we get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalisation of data
x_train /= 255
x_test /= 255

# print('x_train shape:', x_train.shape)
# print('Number of images in x_train', x_train.shape[0])
# print('Number of images in x_test', x_test.shape[0])

model = Sequential()

model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))

# Final Dense layer must have 10 neurons since there are 10 different numbers (0 - 9)
model.add(Dense(10,activation=tf.nn.softmax)) 

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# try:
#     print("Attempting to load model....")
#     model = load_model("model.h5")
# except:
#     print("Unable to load model. Creating new model...")
#     model.fit(x=x_train,y=y_train, epochs=20)
#     model.evaluate(x_test, y_test)
#     model.save("model.h5")

try:
    print("Attempting to load model....")
    model = load_model("mnist_model.h5")
except:
    print("Unable to load model. Creating new model...")
    model.fit(x_train, y_train, epochs=10)
    model.save("mnist_model.h5")

output = model.predict(x_test)

print(model.predict(x_test[100:101]))