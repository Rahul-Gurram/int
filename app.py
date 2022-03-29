import flask
from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
import requests
import pickle

# Use pickle to load in the pre-trained model
with open(f'model/Insurance.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'get':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        age = int(request.form['age'])
        sex = str(request.form['sex'])
        bmi = int(request.form['bmi'])
        children = int(request.form['children'])
        smoker = str(request.form['smoker'])
        region = str(request.form['region'])

        # Make DataFrame for model
        input_variables = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                                       columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'],
                                       index=['input'])

        # Get the model's prediction
        prediction = model.predict(input_variables)
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',
                                     original_input={'Age':age,
                                                     'Sex':sex,
                                                     'BMI':bmi,
                                                     'Children':children,
                                                     'Smoker':smoker,
                                                     'Region':region},
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)