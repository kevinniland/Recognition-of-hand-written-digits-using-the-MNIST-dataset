from flask import Flask, json, jsonify, render_template, request
import keras as kr
from keras.models import load_model
import numpy as np
from model import prediction
import sys

app = Flask(__name__)
model = load_model('model.h5')

@app.route('/')
def homePage():
    return render_template('application/html/frontend.html')

@app.route('/', methods=['GET', 'POST'])
def recogniseImage():
    data = request.get_json()
    value = data['digitArr']

    digitArr = (np.array(value).reshape(1, 784))
    digitInfo = prediction(digitArr, model)

    # print(digitInfo, file = sys.stderr)

    return jsonify({'Digit prediction' : "predPlaceholder"})
    return render_template('application/html/frontend.html')
    
if __name__ == '__main__':
    app.run(debug=True, threaded=False)