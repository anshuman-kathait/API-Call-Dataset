import pandas as pd
import json

# Load top_3_apis from JSON
with open('top_3_apis.json', 'r') as f:
    top_3_apis = json.load(f)

# Specify the dataset file path
file_path = r"C:\Users\katha\OneDrive\Desktop\ML API\API Call Dataset.csv"

# Load the dataset
data = pd.read_csv(file_path)

# Normalize column names
data.columns = data.columns.str.strip().str.lower()

# Filter data for each API and save to CSV
for api in top_3_apis:
    filtered_data = data[data['api code'] == api]
    output_path = f"{api}.csv"
    filtered_data.to_csv(output_path, index=False)
    print(f"Filtered data for {api} saved to {output_path}")
