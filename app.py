import pickle
from flask import Flask, request, app
from flask_cors import CORS
import numpy as np
import pandas as pd
import sklearn

app = Flask(__name__)
CORS(app)

#ROUTE FOR PREDICTION API
@app.route('/predict/api', methods = ['POST'])
def predict_api():
    api_data = request.json['data']
    print(api_data)
    return

if(__name__ == "__main__"):
    app.run(debug = True)