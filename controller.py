# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 07:41:02 2020

@author: Aatish
"""

from flask import Flask, request
from mask_model import MaskModel
from flask_cors import cross_origin

app = Flask(__name__)

@app.route('/getPrediction', methods=['GET', 'POST'])
@cross_origin()
def get_prediction():
    prediction = {'class': int(model.predict_from_base64(str.replace(request.json['imageData'], 'data:image/jpeg;base64,', '')))}
    print(prediction)
    return prediction

model = MaskModel()
model.load_model('model', 'model')
app.run(host='0.0.0.0')