from flask import Flask, jsonify, render_template, request, url_for
import re as reg_expr
import io 
import base64
from PIL import Image
import numpy as np
import tensorflow as tf
#import tensor as ten

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def recogniseImage():
    modelPrediction = 0

    if (request.method == 'POST'):
        imageSize = 28, 28
        imageURL = request.values['imageBase64']
        imageString = reg_expr.search(r'base64, (.*)', imageURL).group(1)
        imageBytes = io.BytesIO(base64.b64decode(imageSize))

        userImage = Image.open(imageBytes) 
        userImage = userImage.resize(imageSize, Image.LANCZOS)  
        userImage = userImage.convert('1') 
        userImageArray = np.asarray(userImage)
        userImageArray = userImageArray.flatten()

        # with tf.Session() as sess:
        #     saver = tf.train.import_meta_graph('model.h5')

        #     predict_number = tf.argmax(ten.y, 1)
        #     predicted_number = ten.sess.run([predict_number], feed_dict={ten.x: [userImageArray]})
        #     modelPrediction = predicted_number[0][0]
        #     modelPrediction = int(modelPrediction)
        #     print(modelPrediction)

        # return jsonify(modelPrediction = modelPrediction)

    return render_template('application/html/frontend.html')
    
if __name__ == '__main__':
    app.run(debug=True)