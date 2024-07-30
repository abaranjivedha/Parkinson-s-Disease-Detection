from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('parkinsons_model_subset.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        form_data = request.form
        features = [float(form_data['PPE']),
                    float(form_data['DFA']),
                    float(form_data['RPDE']),
                    float(form_data['numPulses']),
                    float(form_data['numPeriodsPulses']),
                    float(form_data['meanPeriodPulses']),
                    float(form_data['stdDevPeriodPulses']),
                    float(form_data['locPctJitter'])]
        features = np.array(features).reshape(1, -1)

        # Predict using the loaded model
        prediction = model.predict(features)
        prediction_result = "Parkinson's disease detected" if prediction[0] == 1 else "No Parkinson's disease detected"
        prediction_class = "positive" if prediction[0] == 1 else "negative"
        
        return render_template('index.html', prediction_result=prediction_result, prediction_class=prediction_class)
    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    app.run(debug=True)
