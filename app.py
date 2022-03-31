import flask
from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
import pickle


# Use pickle to load in the pre-trained model
with open(f'model/Insurance.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__)

# Define html file to get user input
@app.route('/')
def home():
    return render_template('main.html')

# prediction function
@app.route('/predict', methods = ['POST'])
def predict(to_predict):
    features = np.array(to_predict).reshape(1, 6)
    results = model.predict(features)
    return render_template("main.html", prediction = results)

# main function
if __name__ == '__main__':
    app.run()
