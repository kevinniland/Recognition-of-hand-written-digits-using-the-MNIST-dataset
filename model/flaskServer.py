# Flask is a popular, extensible web microframework for building web applications with Python. Flask is a 
# web framework which clients make requests to

# flask is the framework while Flask is a Python class datatype

# render_template allows the views i.e the webpages, to be rendered and viewable once the server is up and running

# request contains the data that the client has sent to your appplication, such as the URL parameters, POST data etc.
from flask import Flask, render_template, request

import re
from io import BytesIO

# 
import base64

# Allows for the saving and loading of the model
from keras.models import load_model

# NumPy is the fundamental package for scientific computing with Python
import numpy as np

# PIL (Python Image Library)
from PIL import Image, ImageOps

# OpenCV-Python is a library of Python bindings designed to solve computer vision problems. OpenCV-Python 
# makes use of Numpy, which is a highly optimized library for numerical operations with a MATLAB-style syntax. All 
# the OpenCV array structures are converted to and from Numpy arrays
import cv2

# Creates an instance of the Flask class for the web application. __name__ is a special variable that gets 
# as value the string "__main__" when you’re executing the script
app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')

# Specify the size that the image will be resized to. The MNIST dataset images are centered in a 28 x 28 image so to allow the 
# user-drawn image to be predicted correctly, we resize the image retrieved from the canvas to the same size
imageHeight = 28
imageWidth = 28
size = imageHeight, imageWidth

# Specify the route for the HTML page that conatins the canvas and return the page
@app.route('/')
def homePage():
    return render_template('application/html/frontend.html')

# Specify the route and methods 
@app.route('/digit', methods=['GET', 'POST'])
def recogniseImage():
    imageB64 = request.values.get("imageBase64", "")

    decodeImage = base64.b64decode(imageB64[22:])
    
    with open("userDigit.png", "wb") as f:
        f.write(decodeImage)

    # https://dev.to/preslavrachev/python-resizing-and-fitting-an-image-to-an-exact-size-13ic
    #
    # Open the canvas image sent to the server then, using ImageOps, take it, resize it, and apply a high-quality 
    # downsampling filter (ANTIALIAS). Aliasing in images is described as jagged lines/edges (think of a staircase).
    # Applying anti-aliasing to an image diminshes/resolves this. It applies a particular technique to smooth out the
    # edges for a better overall picture
    originalImage = Image.open("userDigit.png")
    newImage = ImageOps.fit(originalImage, size, Image.ANTIALIAS)

    # Save the resized image, allowing it to be further modified
    newImage.save("resizedUserDigit.png")

    # cv2.imread() loads image from the specified file
    cv2Image = cv2.imread("resizedUserDigit.png")

    # cv2.cvtColor() converts an image from one color space to another. The original black and white (bilevel) 
    # images from NIST were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio. The 
    # resulting images contain grey levels as a result of the anti-aliasing technique used by the normalization 
    # algorithm. As a result, we convert the user-drawn digit to a grey image as well
    grayImage = cv2.cvtColor(cv2Image, cv2.COLOR_BGR2GRAY)

    grayArray = np.array(grayImage).reshape(1, 28, 28, 1)

    setPrediction = model.predict(grayArray)
    getPrediction = np.array(setPrediction[0])

    # To return this as a response, it must be cast as a string (unable to return an int64 in this type of function)
    predictedNumber = str(np.argmax(getPrediction))
    print(predictedNumber)

    # Return the predicted number. The predicted number will then be sent back to the javascript file, where it is
    # then displayed on webpage itself
    return predictedNumber

if __name__ == '__main__':
    app.run(debug=False, threaded=False)