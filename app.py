import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# Load Saved Model
with open("savedModels/model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the indices of the relevant features used during training
relevant_indices = [0, 4, 5, 6, 7, 8, 9, 10, 11, 22]

# Route for Prediction API
@app.route('/predict/api', methods=['POST'])
def predict_api():
    # Extract data from the request
    api_data = request.json['data']

    # Extract relevant features
    relevant_data = [api_data[i] for i in relevant_indices]

    # Make prediction
    prediction = model.predict([relevant_data])

    # Return prediction as JSON response
    return jsonify({'prediction': prediction.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
