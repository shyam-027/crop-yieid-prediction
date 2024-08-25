from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and label encoders
best_model = joblib.load('best_model.joblib')
label_encoders = joblib.load('label_encoders (2).joblib')

# Define crop types for selection
CROP_TYPES = [
    'Arecanut', 'Arhar/Tur', 'Castor seed', 'Coconut ', 'Cotton(lint)', 'Dry chillies', 'Gram', 'Jute',
    'Linseed', 'Maize', 'Mesta', 'Niger seed', 'Onion', 'Other  Rabi pulses', 'Potato',
    'Rapeseed &Mustard', 'Rice', 'Sesamum', 'Small millets', 'Sugarcane', 'Sweet potato', 'Tapioca',
    'Tobacco', 'Turmeric', 'Wheat', 'Bajra', 'Black pepper', 'Cardamom', 'Coriander', 'Garlic', 'Ginger',
    'Groundnut', 'Horse-gram', 'Jowar', 'Ragi', 'Cashewnut', 'Banana', 'Soyabean', 'Barley', 'Khesari',
    'Masoor', 'Moong(Green Gram)', 'Other Kharif pulses', 'Safflower', 'Sannhamp', 'Sunflower', 'Urad',
    'Peas & beans (Pulses)', 'other oilseeds', 'Other Cereals', 'Cowpea(Lobia)', 'Oilseeds total', 'Guar seed',
    'Other Summer Pulses', 'Moth'
]

@app.route('/')
def index():
    return render_template('index.html', crop_types=CROP_TYPES)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    crop = data['crop']
    area = float(data['area'])
    production = float(data['production'])
    annual_rainfall = float(data['annual_rainfall'])
    fertilizer = float(data['fertilizer'])
    pesticide = float(data['pesticide'])

    # Encode the crop type
    crop_encoded = label_encoders['Crop'].transform([crop])[0]

    # Prepare the input for prediction
    input_features = np.array([[crop_encoded, area, production, annual_rainfall, fertilizer, pesticide]])

    # Predict
    prediction = best_model.predict(input_features)[0]

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)