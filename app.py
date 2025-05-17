from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

with open('carinsurance_xgb_clas.pkl', 'rb') as f:
    model = pickle.load(f)

with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

valid_area_clusters = ['C1','C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
                       'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22']

valid_segments = ['A','B1', 'B2', 'C1', 'C2', 'Utility']

valid_models = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11']

valid_engine_types = ['F8D Petrol Engine', '1.2 L K12N Dualjet', '1.0 SCe',
                      '1.5 L U2 CRDi', '1.5 Turbocharged Revotorq', 'K Series Dual jet',
                      '1.2 L K Series Engine', 'K10C', 'i-DTEC', 'G12B',
                      '1.5 Turbocharged Revotron']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form = request.form

        if form['area_cluster'] not in valid_area_clusters:
            return render_template("index.html", prediction_text=f"Error: Invalid area_cluster '{form['area_cluster']}'")

        if form['segment'] not in valid_segments:
            return render_template("index.html", prediction_text=f"Error: Invalid segment '{form['segment']}'")

        if form['model'] not in valid_models:
            return render_template("index.html", prediction_text=f"Error: Invalid model '{form['model']}'")

        if form['engine_type'] not in valid_engine_types:
            return render_template("index.html", prediction_text=f"Error: Invalid engine_type '{form['engine_type']}'")

        tenure_months = float(form['policy_tenure'])
        policy_tenure = round(tenure_months / 12, 6)

        input_dict = {
            'policy_tenure': policy_tenure,
            'age_of_car': float(form['age_of_car']),
            'age_of_policyholder': float(form['age_of_policyholder']),
            'population_density': float(form['population_density']),
            'is_parking_sensors': int(form['is_parking_sensors']),
            'is_power_steering': int(form['is_power_steering']),
            'is_day_night_rear_view_mirror': int(form['is_day_night_rear_view_mirror']),
            'is_speed_alert': int(form['is_speed_alert']),
            'area_cluster': form['area_cluster'],
            'segment': form['segment'],
            'model': form['model'],
            'engine_type': form['engine_type']
        }

        df = pd.DataFrame([input_dict])
        X_transformed = preprocessor.transform(df)
        prediction = model.predict(X_transformed)[0]
        result = "CLAIM" if prediction == 1 else "NO CLAIM"
        return render_template("index.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")
    
    
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

