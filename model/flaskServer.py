# Flask is a popular, extensible web microframework for building web applications with Python. Flask is a 
# web framework which clients make requests to

# flask is the framework while Flask is a Python class datatype

# render_template allows the views i.e the webpages, to be rendered and viewable once the server is up and running

# request contains the data that the client has sent to your appplication, such as the URL parameters, POST data etc.
from flask import Flask, render_template, request

# io provides Python’s main facilities for dealing with various types of I/O. From this module, we are importing BytesIO

# BytesIO is a stream implementation using an in-memory bytes buffer. BytesIO provides or overrides these methods in 
# addition to those from BufferedIOBase and IOBase
from io import BytesIO

# base64 provides functions for encoding binary data to printable ASCII characters and decoding such encodings 
# back to binary data
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

@app.route('/digit', methods=['GET', 'POST'])
def recogniseImage():
    imageB64 = request.values.get("imageBase64", "")

    # Decode the image
    decodeImage = base64.b64decode(imageB64[22:])

    # .convert('L') converts the image to black and white
    userImage = Image.open(BytesIO(decodeImage)).convert('L')
    userImage.save("userDigit.png")

    # Resize the image to the same size of the images in the MNIST dataset and apply anti-aliasing to it.
    # Aliasing in images is described as jagged lines/edges (think of a staircase).
    # Applying anti-aliasing to an image diminshes/resolves this. It applies a particular technique to smooth out the
    # edges for a better overall picture
    userImage = userImage.resize(size, Image.ANTIALIAS)

    ''' Previous method I used. While this way worked pretty well (using ImageOps), 1's, 6's, and 9's were rarely getting 
    detected. As a result, I have implemented a different method using the Python Imaging Library (PIL) '''   

    #   Save the canvas image as a .png file
    #   with open("userDigit.png", "wb") as f:
    #       f.write(decodeImage)

    # See https://dev.to/preslavrachev/python-resizing-and-fitting-an-image-to-an-exact-size-13ic
    #
    # Open the canvas image sent to the server then, using ImageOps, take it, resize it, and apply a high-quality 
    # downsampling filter (ANTIALIAS). Aliasing in images is described as jagged lines/edges (think of a staircase).
    # Applying anti-aliasing to an image diminshes/resolves this. It applies a particular technique to smooth out the
    # edges for a better overall picture
    # originalImage = Image.open("userDigit.png")
    # newImage = ImageOps.fit(originalImage, size, Image.ANTIALIAS)

    # Save the resized image, allowing it to be further modified
    # newImage.save("resizedUserDigit.png")

    # cv2.imread() loads image from the specified file
    # cv2Image = cv2.imread("resizedUserDigit.png")

    # cv2.cvtColor() converts an image from one color space to another. The original black and white (bilevel) 
    # images from NIST were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio. The 
    # resulting images contain grey levels as a result of the anti-aliasing technique used by the normalization 
    # algorithm. As a result, we convert the user-drawn digit to a grey image as well

    # grayImage = cv2.cvtColor(cv2Image, cv2.COLOR_BGR2GRAY)

    # Reshape the gray image using a numpy array
    # grayArray = np.array(grayImage).reshape(1, 28, 28, 1)

    # See https://www.geeksforgeeks.org/python-pil-image-point-method/

    # Thresholding is a type of image segmentation, where we change the pixels of an image to make the image easier to 
    # analyze. In thresholding, we convert an image from color or grayscale into a binary image, i.e., one that is simply 
    # black and white. Most frequently, we use thresholding as a way to select areas of interest of an image, 
    # while ignoring the parts we are not concerned with. If you open up resizedUserDigit.png, you can see that the image
    # now looks like the images in the MNIST dataset
    threshold = 0

    # .point() maps this image through a lookup table or function
    userImage = userImage.point(lambda p: p > threshold and 255) 
    userImage.save("resizedUserDigit.png")

    # Flatten the numpy array and reshape it. Due to the way I wrote my model, my reshape method looks
    grayArray = np.ndarray.flatten(np.array(userImage)).reshape(1, 28, 28, 1).astype("uint8") / 255

    # setPrediction = model.predict(grayArray)
    # getPrediction = np.array(setPrediction[0])

    # Pass the array into the model for prediction (Shorter version of above two lines)
    getPrediction = np.array(model.predict(grayArray)[0])

    # To return this as a response, it must be cast as a string (unable to return an int64 in this type of function)
    predictedNumber = str(np.argmax(getPrediction))
    print(predictedNumber)
    
    # Return the predicted number. The predicted number will then be sent back to the javascript file, where it is
    # then displayed on webpage itself
    return predictedNumber
      
# For some reason, I kept getting an error when attempting to run the server if debug and threaded was set to True. I kept 
# getting the following error: 
# 
# AttributeError: '_thread._local' object has no attribute 'value'. 
# 
# From looking around online, it was recommended that I set to debug and threaded to false, which did fix the issue
if __name__ == '__main__':
    app.run(debug=False, threaded=False)