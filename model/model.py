# Imports
import matplotlib.pyplot as plt
import numpy as np

# Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano
import keras
from keras.datasets import mnist # Import the MNIST dataset directly from the Keras API
from keras.models import Sequential # The Sequential model is a linear stack of layers
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import load_model

(train_imgs, train_labels), (test_imgs, test_labels) = mnist.load_data()

# Defines the size of the image as 28 * 28 pixels
img_rows, img_cols = 28, 28

# Reshaping
#
# To be able to use the MNIST dataset with the Keras API, we need to change our array (which is 3-dimensional)
# to 4-dimensional numpy arrays. We also must 'normalize' our data, as is always required in neural networks.
# This can be done by dividing the RGB codes of the images to 255
# if k.image_data_format() == 'channels_first':
#     train_imgs = train_imgs.reshape(train_imgs.shape[0], 1, img_rows, img_cols)
#     test_imgs = test_imgs.reshape(test_imgs.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
train_imgs = train_imgs.reshape(train_imgs.shape[0], img_rows, img_cols, 1)
test_imgs = test_imgs.reshape(test_imgs.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# Ensure the values of train_imgs and test_imgs are float. This is done so as we can get decimal points after division
train_imgs = train_imgs.astype('float32')
test_imgs = test_imgs.astype('float32')

train_imgs /= 255
test_imgs /= 255

print(train_imgs.shape[0], "training samples")
print(test_imgs.shape[0], "testing samples")

# In the model, we can experiment with any number for the first Dense layer. However, the final Dense layer must have
# 10 neurons since there are 10 number classes (0, 1, 2, 3, ..., 9)
num_classes = 10

train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)
train_labels[0]

# Create a Sequential model and add the layers
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())  # Flattens the 2D arrays for fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

# Determines the number of samples that will be propagated through the neural network
batch_size = 128 

# In the context of a neural network, one epoch is the equivalent of one forward pass and one backward pass of
# all the training examples
num_epoch = 100

# To avoid having to train the model each time the program is ran, the trained model can be loaded from a file
# If no file is created, then the model is trained and then saved to a file
try:
    print("Attempting to load model....")
    model = load_model("model_digit.h5")
except:
    print("Failed to load model. Creating new model...")
    model_log = model.fit(train_imgs, train_labels,
                          batch_size=batch_size,
                          epochs=num_epoch,
                          verbose=1,
                          validation_data=(test_imgs, test_labels))

    model.save_weights("model_digit.h5")

    model.save("model_digit.h5")
    print("Saved model. Model will now be loaded on next run through")

plt.imshow(test_imgs[999].reshape(28, 28), cmap="gray")
plt.show()

print(model.predict(test_imgs[999:1000]))
