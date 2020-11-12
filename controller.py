# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 07:41:02 2020

@author: Aatish
"""

from flask import Flask, request
from mask_model import MaskModel

app = Flask(__name__)

@app.route('/getPrediction/')
def get_prediction():
    prediction = {'class': int(model.predict_from_base64(request.json['image']))}
    print(prediction)
    return prediction

model = MaskModel()
model.load_model('model', 'model')
app.run()