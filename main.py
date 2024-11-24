'''(1)from flask import Flask, request, jsonify
import pickle
import joblib  # For loading models
import pandas as pd  # For data handling

# Initialize Flask app
app = Flask(__name__)'''

from flask import Flask, request, jsonify, render_template
import joblib  # For loading models
import pandas as pd  # For data handling

# Initialize Flask app
app = Flask(__name__)
# Load your model (ensure the model file is in the working directory)
model = joblib.load("arima_model.pkl")  # Update with actual filename
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract year and month from the form
        year = int(request.form.get("year"))
        month = int(request.form.get("month"))

        # Create a Timestamp for the forecast date
        forecast_date = pd.Timestamp(year=year, month=month, day=1)

        # Retrieve the last date from the model's training data
        last_date = pd.Timestamp("2020-12-01")  # Replace with the actual last date if needed

        # Calculate steps for ARIMA forecasting
        steps = (forecast_date.year - last_date.year) * 12 + (forecast_date.month - last_date.month)

        # Ensure steps are positive
        if steps <= 0:
            return render_template('index.html', prediction_text="Error: Forecast date must be after the model's last training date.")

        # Make prediction
        prediction = model.forecast(steps=steps)[-1]

        # Render the result on the page
        return render_template('index.html', prediction_text=f"Prediction: {prediction}")
    except Exception as e:
        # Handle errors gracefully
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)


'''(1)@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        input_data = request.json

        # Extract year and month from input
        year = input_data["year"]
        month = input_data["month"]

        # Create a Timestamp for the forecast date
        forecast_date = pd.Timestamp(year=year, month=month, day=1)

        # Retrieve last date (manually specify if the model's training data is not stored)
        # Replace this with the actual date format or value used during training
        if hasattr(model.data, 'endog') and hasattr(model.data.endog, 'index'):
            last_date = model.data.endog.index[-1]
        else:
            # Specify the last known date if not automatically accessible
            last_date = pd.Timestamp("2024-01-01")  # Replace with your model's last training date

        # Calculate steps for ARIMA forecasting
        steps = (forecast_date.year - last_date.year) * 12 + (forecast_date.month - last_date.month)

        # Ensure steps are positive
        if steps <= 0:
            return jsonify({"error": "Forecast date must be after the model's last known date."}), 400

        # Make prediction
        prediction = model.forecast(steps=steps)[-1]

        # Return the prediction
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400



if __name__ == '__main__':
    app.run(debug=True)'''
