from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained waste generation forecasting model
with open('waste_generation_forecasting_model.pkl', 'rb') as file:
    waste_gen_model = pickle.load(file)

# Sample data for insights and route optimization (replace with real data as needed)
sample_waste_data = pd.DataFrame({
    'Location': ['Zone A', 'Zone B', 'Zone C', 'Zone D'],
    'Average Distance': [5, 10, 15, 20],  # Example distances in km
    'Waste Generated': [100, 200, 150, 180]  # Example past data in kg
})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_waste', methods=['POST'])
def predict_waste():
    try:
        location = request.form['location']
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        day_of_week = request.form['day_of_week']

        # Mapping input values for model
        day_of_week_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        location_map = {'Zone A': 0, 'Zone B': 1, 'Zone C': 2, 'Zone D': 3}

        day_of_week_val = day_of_week_map.get(day_of_week, -1)
        location_val = location_map.get(location, -1)

        if day_of_week_val == -1 or location_val == -1:
            return jsonify({"error": "Invalid location or day of week."}), 400

        # Prepare input data and make prediction
        input_data = np.array([[temperature, humidity, day_of_week_val, location_val]])
        input_df = pd.DataFrame(input_data, columns=["Temperature", "Humidity", "DayOfWeek", "Location"])
        predicted_waste = waste_gen_model.predict(input_df)

        return jsonify({"predicted_waste": f"{predicted_waste[0]:.2f} kg"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/optimize_route', methods=['POST'])
def optimize_route():
    location = request.form['location']
    if location in sample_waste_data['Location'].values:
        distance = sample_waste_data[sample_waste_data['Location'] == location]['Average Distance'].values[0]
        return jsonify({"optimized_route": f"{location} with distance {distance} km"})
    else:
        return jsonify({"error": "Invalid location."}), 400

@app.route('/get_insights', methods=['GET'])
def get_insights():
    # Calculate basic insights from sample data
    avg_waste = sample_waste_data['Waste Generated'].mean()
    max_waste = sample_waste_data['Waste Generated'].max()
    min_waste = sample_waste_data['Waste Generated'].min()

    insights = {
        "avg_waste": f"{avg_waste:.2f} kg",
        "max_waste": f"{max_waste:.2f} kg",
        "min_waste": f"{min_waste:.2f} kg"
    }
    return jsonify(insights)

if __name__ == '__main__':
    app.run(debug=True)
