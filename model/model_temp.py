# Adapted from: https://docs.python.org/3/library/gzip.html

import gzip
import numpy as np

# Import keras
import keras as kr
from keras.models import load_model


#%matplotlib inline
import matplotlib.pyplot as plt

# For encoding categorical variables
import sklearn.preprocessing as pre

with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    file_content = f.read()

with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    labels = f.read()

print(type(file_content))
file_content[0:4]

previous_image = 16
next_image = 800
label_index = 8

# Start a neural network, building it by layers
model = kr.models.Sequential()

# Add a hidden layer with 1000 neurons and an input layer with 784
model.add(kr.layers.Dense(units=600, activation='linear', input_dim=784))
model.add(kr.layers.Dense(units=400, activation='relu'))

# Add a three neuron output layer.
model.add(kr.layers.Dense(units=10, activation='softmax'))

# Build the graph
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
    train_img = f.read()

with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
    train_lbl = f.read()

train_img = ~np.array(list(train_img[16:])).reshape(60000, 28, 28).astype(np.uint8) / 255.0
train_lbl =  np.array(list(train_lbl[ 8:])).astype(np.uint8)

inputs = train_img.reshape(60000, 784)

encoder = pre.LabelBinarizer()
encoder.fit(train_lbl)
outputs = encoder.transform(train_lbl)

try:
    print("Attempting to load model....")
    model = load_model("model.h5")
except:
    print("Unable to load model. Creating new model...")
    model.fit(inputs, outputs, epochs=10, batch_size=100)
    model.save("model.h5")

with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_img = f.read()

with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    test_lbl = f.read()
    
test_img = ~np.array(list(test_img[16:])).reshape(10000, 784).astype(np.uint8) / 255.0
test_lbl =  np.array(list(test_lbl[8:])).astype(np.uint8)

output = model.predict(test_img)

print((encoder.inverse_transform(model.predict(test_img)) == test_lbl).sum())

print(model.predict(test_img[100:101]))

plt.imshow(test_img[100].reshape(28, 28), cmap="gray")
plt.show()