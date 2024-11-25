from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("arima_model.pkl")
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        year = int(request.form.get("year"))
        month = int(request.form.get("month"))

        forecast_date = pd.Timestamp(year=year, month=month, day=1)
        last_date = pd.Timestamp("2020-12-01")

        steps = (forecast_date.year - last_date.year) * 12 + (forecast_date.month - last_date.month)

        if steps <= 0:
            return jsonify({"error": "Forecast date must be after the model's last training date."}), 400

        prediction = model.forecast(steps=steps)[-1]

        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)



