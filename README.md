# Recognition of hand-written digits using the MNIST dataset
 _4th year Emerging Technologies project. This project concerns the well-known MNIST dataset and the Python packages keras, ﬂask, and jupyter. This application contains a model that recognises hand-written digits using the MNIST dataset and a web application that allows the user to draw a digit using their mouse or touchscreen device. This application was written mainly in Python._
 
 ## Purpose of project
 This project serves as an introduction to machine learning. [MNIST](http://yann.lecun.com/exdb/mnist/) (Modified National Institute of
 Standards and Technology) is the de facto "Hello World" dataset of computer vision. Released in 1999, this classic dataset of
 handwritten images has served as the basis for benchmarking classification algorithms.
 
 ## How to run
 * Clone or download the project folder. To clone, use the command: `git clone https://github.com/kevinniland97/Recognition-of-hand-written-digits-using-the-MNIST-dataset`
 * A trained model has already been provided: `model.h5`. To train the model yourself, simply run `model.py` using `py model.py`.
 * To run the flask server, run `flaskServer.py` using `py flaskServer.py`.
 * Once the server is up and running, the server will be running at `http://127.0.0.1:5000/ `. Press `Ctrl` and click on it to follow the link or copy and paste it into your desired browser's address bar.
 * This will open up a webpage containing a canvas. Draw any digit (from 0 - 9) and click `Save`. This will send your digit to the flask server. From here, the image will be resized to the proper size (28 x 28) and then put through the model. The image will then be compared against the dataset which the model was trained against. The model will then predict what digit you have drawn and output the result to the webpage and to the python console.
 
 ## Project Layout
 This project contains three distinct "parts" that all work together:
 * **Model**: The model has been trained using the MNIST dataset. The dataset contains 70,000 handwritten digits, split into two sets - a training set (60,000) and a testing set (10,000). It is a subset of the larger NIST set.
 * **Flask server**: The flask server runs the entire application. When run, the application will be hosted on `http://127.0.0.1:5000/`. The flask server displays the canvas and processess the canvas image that is sent from the canvas using AJAX and puts it through the model.
 * **Canvas**: The canvas is made up of three parts - the HTML file that renders the actual canvas, the JavaScript file that contains functions that allows the user to use their mouse on the canvas and send the canvas image to the flask server to processed and put through the model, and the CSS file that includes some styling.

## Project flow
The general flow of the project is as follows:
* Run the flask server.
* Open the webpage.
* Use your mouse to draw a digit (0 - 9) on the canvas and click `Submit`.
* The canvas image is submitted to the flask server via a POST request using AJAX.
* The image is received on the flask server and processed - decode the image (remove unnecessary information from the image), convert the image to black and white and resize it, put the image through the model for prediction.
* Send the result of the prediction back and display it on the webpage.

![Project Flow diagram](https://github.com/kevinniland97/Recognition-of-hand-written-digits-using-the-MNIST-dataset/blob/master/project_flow_diagram.PNG)
