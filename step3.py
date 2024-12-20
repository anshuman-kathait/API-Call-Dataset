import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import json

# Load the list of top 3 APIs
with open("top_3_apis.json", "r") as file:
    top_3_apis = json.load(file)

# Function to process and extract features
def process_dataset(file_path):
    data = pd.read_csv(file_path)
    # Normalize column names
    data.columns = data.columns.str.strip().str.lower()

    # Convert 'time of call' to datetime
    data['time of call'] = pd.to_datetime(data['time of call'], format='%d-%m-%Y %H:%M')

    # Extract features
    data['hour'] = data['time of call'].dt.hour
    data['day'] = data['time of call'].dt.day
    data['day_of_week'] = data['time of call'].dt.dayofweek
    data['time_since_first_call'] = (data['time of call'] - data['time of call'].min()).dt.total_seconds()

    # Drop the original 'time of call'
    data = data.drop(columns=['time of call'])

    return data

# Train models for each API
for api in top_3_apis:
    csv_file = f"{api}.csv"
    if not os.path.exists(csv_file):
        print(f"File {csv_file} not found, skipping.")
        continue

    # Process dataset
    data = process_dataset(csv_file)
    features = data.drop(columns=['api code'])  # Corrected column name
    target = features.pop('time_since_first_call')

    # Check for empty features
    if features.empty:
        print(f"No valid features in {csv_file}, skipping.")
        continue

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train and evaluate models
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor()
    }

    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            print(f"{model_name} for {api}: MSE = {mse:.2f}")
            joblib.dump(model, f"{api}_{model_name}.joblib")
        except Exception as e:
            print(f"Error training {model_name} for {api}: {e}")


# Store the best model for each API
best_models = {}

for api in top_3_apis:
    csv_file = f"{api}.csv"
    if not os.path.exists(csv_file):
        continue

    # Load models and compare MSE
    best_model_name = None
    best_mse = float('inf')

    for model_name in ["LinearRegression", "RandomForest"]:
        model_file = f"{api}_{model_name}.joblib"
        if os.path.exists(model_file):
            try:
                model = joblib.load(model_file)
                data = process_dataset(csv_file)
                features = data.drop(columns=['api code'])
                target = features.pop('time_since_first_call')

                # Dummy evaluation to find the best model
                _, X_test, _, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
                predictions = model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)

                if mse < best_mse:
                    best_mse = mse
                    best_model_name = model_file
            except Exception as e:
                print(f"Error evaluating {model_name} for {api}: {e}")

    if best_model_name:
        best_models[api] = {"model": best_model_name, "mse": best_mse}

# Save the best models to a JSON file
with open("best_models.json", "w") as file:
    json.dump(best_models, file, indent=4)

print("Best models saved to best_models.json")
