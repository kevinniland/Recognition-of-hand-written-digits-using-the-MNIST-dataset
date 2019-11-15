# Recognition of hand-written digits using the MNIST dataset
 _4th year Emerging Technologies project. This project concerns the well-known MNIST dataset and the Python packages keras, ﬂask, and jupyter. This application contains a model that recognises hand-written digits using the MNIST dataset and a web application that allows the user to draw a digit using their mouse or touchscreen device. This application was written mainly in Python._
 
 ## Purpose of project
 
 ## How to run
 * Clone or download the project folder. To clone, use the command: `git clone https://github.com/kevinniland97/Recognition-of-hand-written-digits-using-the-MNIST-dataset`
 * A trained model has already been provided: `model.h5`. To train the model yourself, simply run `model.py` using `py model.py`.
 * To run the flask server, run `flaskServer.py` using `py flaskServer.py`.
 * Once the server is up and running, the server will be running at `http://127.0.0.1:5000/ `. Press `Ctrl` and click on it to follow the link or copy and paste it into your desired browser's address bar.
 * This will open up a webpage containing a canvas. Draw any digit (from 0 - 9) and click `Save`. This will send your digit to the flask server. From here, the image will be resized to the proper size (28 x 28) and then put through the model. The image will then be compared against the dataset which the model was trained against. The model will then predict what digit you have drawn and output the result to the webpage and to the python console.
