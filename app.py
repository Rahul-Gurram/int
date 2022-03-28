import flask
import pickle
import pandas as pd

# Use pickle to load in the pre-trained model
with open(f'model/Insurance.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        age = flask.request.form['age']
        sex = flask.request.form['sex']
        bmi = flask.request.form['bmi']
        children = flask.request.form['children']
        smoker = flask.request.form['smoker']
        region = flask.request.form['region']

        # Make DataFrame for model
        input_variables = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                                       columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'],
                                       index=['input'])

        # Get the model's prediction
        prediction = model.predict(input_variables)[0]
    
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