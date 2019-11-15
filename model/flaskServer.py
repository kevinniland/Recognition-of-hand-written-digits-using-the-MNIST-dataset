from flask import Flask, json, jsonify, render_template, request
import re
import base64
import keras as kr
from keras.models import load_model
import numpy as np
from model import prediction
from PIL import Image, ImageOps
import io
import cv2
from io import StringIO
from io import BytesIO

app = Flask(__name__)
model = load_model('model.h5')

imageHeight = 28
imageWidth = 28

size = imageHeight, imageWidth

@app.route('/')
def homePage():
    return render_template('application/html/frontend.html')

# @app.route('/digit', methods=['POST'])
# def recogniseImage():
#     if request.method == 'POST':
#         imageB64 = request.values['imageBase64']

#         imageData = re.sub('^data:image/.+;base64,', '', imageB64)

#         decodeImage = base64.b64decode(imageData)

#         decodeImage = ~np.array(list(decodeImage)).astype(np.uint8) / 255.0
#         decodeImage.resize(1, 28, 28, 1)

#         # image = Image.open(BytesIO(decodeImage))
#         # image = image.save("userImage.png")
#         # cv2Image = cv2.imread("userImage.png")
#         # grayImage = cv2.cvtColor(cv2Image, cv2.COLOR_BGR2GRAY)

#         # grayArray = np.ndarray.flatten(np.array((grayImage)).reshape(1, 28, 28, 1).astype(np.uint8)) / 255.0

#         setPrediction = model.predict(decodeImage)
#         # setPrediction = model.predict(grayArray)
#         getPrediction = np.array(setPrediction[0])

#     print(getPrediction)
#     predictedNumber = np.argmax(getPrediction)
#     print(predictedNumber)

#     # print(predictedNumber)
#     # imageData = request.json['imageData']
#     # imageNumpyArray = (np.array(imageData).reshape(1, 28, 28, 1))
#     # predictedNumber = prediction(imageNumpyArray, model)
#     # print(imageNumpyArray)
#     # return jsonify(int(predictedNumber))

#     return render_template('application/html/frontend.html')

@app.route('/digit', methods=['GET', 'POST'])
def recogniseImage():
    imageB64 = request.values.get("imageBase64", "")

    decodeImage = base64.b64decode(imageB64[22:])
    
    with open("userDigit.png", "wb") as f:
        f.write(decodeImage)

    originalImage = Image.open("userDigit.png")
    newImage = ImageOps.fit(originalImage, size, Image.ANTIALIAS)

    newImage.save("resizedUserDigit.png")

    cv2Image = cv2.imread("resizedUserDigit.png")
    grayImage = cv2.cvtColor(cv2Image, cv2.COLOR_BGR2GRAY)

    grayArray = (np.array(grayImage).reshape(1, 28, 28, 1))

    setPrediction = model.predict(grayArray)
    getPrediction = np.array(setPrediction[0])

    # print(getPrediction)
    predictedNumber = np.argmax(getPrediction)
    print(predictedNumber)

    return {"serverMessage": imageB64}

if __name__ == '__main__':
    app.run(debug=False, threaded=False)
