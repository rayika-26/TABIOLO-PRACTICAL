# INSTALL AND IMPORT PACKAGES needed.

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# LOAD the SVM model and SCALER model.
loaded_model = joblib.load('svm_model.pkl')
sc = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form.get("age"))
        hypertension = float(request.form.get("hypertension"))
        heart_disease = float(request.form.get("heart_disease"))
        bmi = float(request.form.get("bmi"))
        hba1c_level = float(request.form.get("hba1c_level"))
        blood_glucose_level = float(request.form.get("blood_glucose_level"))

        user_input = np.array([age, hypertension, heart_disease, bmi, hba1c_level, blood_glucose_level]).reshape(1, -1)

        user_input_scaled = sc.transform(user_input)

        prediction = loaded_model.predict(user_input_scaled)

        if prediction == 1:
            result = "The user has Diabetes."
        else:
            result = "The user doesn't have Diabetes."

        return render_template('index.html', prediction_result=result)
    except Exception as e:
        return render_template('index.html', prediction_result="Error: Please check your input.")

if __name__ == '__main__':
    app.run(debug=True)