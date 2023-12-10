from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('diabetes_prediction_model.pkl')

# Column names
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            features = [
                int(request.form['pregnancies']),
                int(request.form['glucose']),
                int(request.form['blood_pressure']),
                int(request.form['skin_thickness']),
                int(request.form['insulin']),
                float(request.form['bmi']),
                float(request.form['diabetes_pedigree_function']),
                int(request.form['age'])
            ]

            # Check if all features are provided
            if None in features:
                raise ValueError("All features must be provided")

            # Make a prediction using the loaded model and feature names
            prediction = model.predict(pd.DataFrame([features], columns=feature_names))[0]

            # Display the result on the same page
            return render_template('predict.html', result=prediction)

        except Exception as e:
            # Log the error using Flask's logger
            app.logger.error(f"An error occurred: {e}")
            return render_template('predict.html', result="Error occurred")

    # Return a proper HTTP response code for Method Not Allowed
    return render_template('predict.html', result="Method Not Allowed"), 405

if __name__ == '__main__':
    app.run(debug=True)
