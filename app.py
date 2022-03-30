import flask
from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
import pickle

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 12)
    loaded_model = pickle.load(open(f'model/Insurance.pkl', "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]
 
@app.route('/', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)       
        return render_template("main.html", prediction = result)

