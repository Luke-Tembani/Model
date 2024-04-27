import pickle
import numpy as np

# Load the saved model
with open("savedModels/model_dt.pkl", "rb") as f:
    model = pickle.load(f)

# Prepare input data for prediction
# Example input data (modify this according to your actual input data)
input_data = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],   # Example numerical features
                       ['tcp', 'ftp_data', 'SF'],       # Example categorical features
                       [10, 11, 12]])                   # Example additional features

# Preprocess input data as needed (scaling, encoding, etc.)
# Ensure that the input data is in the same format as used during training

# Make predictions
predictions = model.predict(input_data)

# Print predictions
print("Predictions:", predictions)
