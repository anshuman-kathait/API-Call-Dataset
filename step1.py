import pandas as pd
import json

# Specify the dataset file path
file_path = r"C:\Users\katha\OneDrive\Desktop\ML API\API Call Dataset.csv"

# Load the dataset
data = pd.read_csv(file_path)

# Normalize column names
data.columns = data.columns.str.strip().str.lower()

# Count frequency of API calls using the corrected column name
api_frequency = data['api code'].value_counts()

# Top 3 most called APIs
top_3_apis = api_frequency.head(3).index.tolist()
print(f"Top 3 APIs: {top_3_apis}")

# Save top_3_apis to a JSON file
with open('top_3_apis.json', 'w') as f:
    json.dump(top_3_apis, f)
print("Top 3 APIs saved to top_3_apis.json")
