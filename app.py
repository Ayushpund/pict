from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import folium
import webbrowser
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Simulated soil dataset for training
np.random.seed(42)
data = {
    'N': np.random.normal(120, 30, 500),
    'P': np.random.normal(50, 15, 500),
    'K': np.random.normal(80, 20, 500),
    'pH': np.random.normal(6.5, 0.8, 500),
    'SOC': np.random.uniform(0.5, 4.0, 500)
}
df = pd.DataFrame(data)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(df[['N', 'P', 'K', 'pH']])
y = df['SOC']

# Train the model
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X, y)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return jsonify({
            "message": "Welcome to the Soil Organic Carbon Prediction API. Use POST to send soil data for prediction."
        })

    try:
        # Get data from request body (form-data)
        N = float(request.form.get("N"))
        P = float(request.form.get("P"))
        K = float(request.form.get("K"))
        pH = float(request.form.get("pH"))
        latitude = float(request.form.get("latitude"))
        longitude = float(request.form.get("longitude"))

        # Scale input data
        user_input_scaled = scaler.transform([[N, P, K, pH]])
        predicted_soc = model.predict(user_input_scaled)[0]

        # Determine SOC status
        if predicted_soc > 3.0:
            soc_status = "Very Good (Rich in Organic Carbon)"
            color = "darkgreen"
        elif 2.0 <= predicted_soc <= 3.0:
            soc_status = "Good (Healthy Soil)"
            color = "green"
        elif 1.0 <= predicted_soc < 2.0:
            soc_status = "Moderate (Needs Improvement)"
            color = "orange"
        else:
            soc_status = "Poor (Low Soil Organic Carbon)"
            color = "red"

        # Generate map
        soil_map = folium.Map(location=[latitude, longitude], zoom_start=12)
        folium.Marker(
            [latitude, longitude],
            popup=f"<b>Predicted SOC:</b> {predicted_soc:.4f} <br> <b>Status:</b> {soc_status}",
            icon=folium.Icon(color=color)
        ).add_to(soil_map)

        # Save the map and return response
        map_path = "soil_map.html"
        soil_map.save(map_path)
        webbrowser.open(map_path)

        return jsonify({
            "SOC": predicted_soc,
            "status": soc_status,
            "latitude": latitude,
            "longitude": longitude
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)