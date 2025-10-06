import numpy as np
from flask import Flask, request, jsonify, render_template 
import pickle
import sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
# Load the trained model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print("Input data:", data)
    
    # Convert input data to numpy array
    input_array = np.array(list(data.values())).reshape(1, -1)
    print("Reshaped array:", input_array)
    
    # Scale the input data
    scaled_data = scaler.transform(input_array)
    
    # Make prediction
    output = regmodel.predict(scaled_data)
    print("Predicted price:", output[0])
    
    # Return prediction as JSON
    return jsonify({"prediction": float(output[0])})

if __name__ == '__main__':
    app.run(debug=True)   