from flask import Flask, render_template, request
import pandas as pd
import pickle
print("Starting Flask server...")

app = Flask(__name__)

# Load trained model pipeline
model = pickle.load(open('bangalore_home_prices_model.pkl', 'rb'))

# Load location list for dropdown (optional but nice)
data = pd.read_csv("CLeaned_data.csv")
locations = sorted(data['location'].unique())

@app.route('/')
def home():
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    sqft = float(request.form['sqft'])
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])
    location = request.form['location']

    # Create input DataFrame (must match training format!)
    input_df = pd.DataFrame([[sqft, bath, bhk, location]],
                            columns=['total_sqft', 'bath', 'bhk', 'location'])

    # Predict
    predicted_price = round(model.predict(input_df)[0], 2)

    return render_template('index.html',
                           prediction=predicted_price,
                           locations=locations,
                           selected_location=location,
                           selected_bhk=bhk,
                           selected_bath=bath,
                           selected_sqft=sqft)
if __name__ == '__main__':
    app.run(debug=True)
