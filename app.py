from flask import Flask, render_template, request
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(_name_)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = [float(x) for x in request.form.values()]
        data_scaled = scaler.transform([data])
        prediction = model.predict(data_scaled)
        return render_template('index.html', prediction_text=f'Estimated House Price: ${prediction[0]*1000:.2f}')
    except:
        return render_template('index.html', prediction_text="Invalid input. Please enter numeric values.")

if _name_ == "_main_":
    app.run(debug=True)