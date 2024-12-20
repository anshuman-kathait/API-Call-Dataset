import pandas as pd
import joblib
import json

# Load the best models
with open("best_models.json", "r") as file:
    best_models = json.load(file)

# Deployment function
def deploy_model(api, input_data):
    if api not in best_models:
        return f"No model available for API {api}"

    # Load the model
    model_path = best_models[api]["model"]
    model = joblib.load(model_path)

    # Prepare input data
    input_df = pd.DataFrame([input_data])
    input_df['time of call'] = pd.to_datetime(input_df['time of call'], format='%d-%m-%Y %H:%M')
    input_df['hour'] = input_df['time of call'].dt.hour
    input_df['day'] = input_df['time of call'].dt.day
    input_df['day_of_week'] = input_df['time of call'].dt.dayofweek
    input_df['time_since_first_call'] = (input_df['time of call'] - input_df['time of call'].min()).dt.total_seconds()
    input_df = input_df.drop(columns=['time of call', 'api code'])

    # Ensure the features match those used during training
    input_df = input_df.drop(columns=['time_since_first_call'], errors='ignore')

    # Make predictions
    prediction = model.predict(input_df)
    return prediction[0]

# Example input data
input_data = {
    "api code": "",  # This will be dynamically updated for each API
    "time of call": "28-11-2024 10:30"
}

# Generate predictions for all top 3 APIs
results = {}
for api in best_models.keys():
    input_data["api code"] = api
    prediction = deploy_model(api, input_data)
    results[api] = prediction

# Display results
for api, prediction in results.items():
    print(f"Prediction for API {api}: {prediction}")
