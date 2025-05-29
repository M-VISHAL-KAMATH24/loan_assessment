from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle as pk
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and scaler
model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)  # Debug
        no_of_dependents = int(data['no_of_dependents'])
        education = int(data['education'])
        self_employed = int(data['self_employed'])
        income_annum = float(data['income_annum'])
        loan_amount = float(data['loan_amount'])
        loan_term = int(data['loan_term'])
        cibil_score = int(data['cibil_score'])
        assets = float(data['assets'])

        input_data = pd.DataFrame([[no_of_dependents, education, self_employed, 
                                    income_annum, loan_amount, loan_term, 
                                    cibil_score, assets]], 
                                  columns=['no_of_dependents', 'education', 'self_employed', 
                                           'income_annum', 'loan_amount', 'loan_term', 
                                           'cibil_score', 'Assets'])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        result = 'Approved' if prediction == 1 else 'Rejected'
        return jsonify({'loan_status': result})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    
    