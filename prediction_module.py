# Import necessary libraries
import os
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load match data
data_path = "nepal_male_json"
all_match_data = []

# Iterate through JSON files in the folder
for filename in os.listdir(data_path):
    if filename.endswith(".json"):
        file_path = os.path.join(data_path, filename)
        try:
            with open(file_path, "r") as json_file:
                match_data = json.load(json_file)
                # Extract relevant information
                meta_info = match_data["meta"]
                info = match_data["info"]
                innings = match_data["innings"]
                # Append extracted data to the list
                all_match_data.append(
                    {
                        "meta_info": meta_info,
                        "info": info,
                        "innings": innings,
                    }
                )
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Create a DataFrame
combined_data = pd.DataFrame(all_match_data)

# Data Preprocessing
combined_data["winner"] = combined_data["info"].apply(
    lambda x: x.get("outcome", {}).get("winner", "No Winner")
)
combined_data["venue"] = combined_data["info"].apply(lambda x: x.get("venue", ""))
combined_data["match_type"] = combined_data["info"].apply(
    lambda x: x.get("match_type", "")
)

# Encode categorical features
label_encoder = LabelEncoder()
combined_data["encoded_venue"] = label_encoder.fit_transform(combined_data["venue"])
combined_data["encoded_match_type"] = label_encoder.fit_transform(
    combined_data["match_type"]
)


# Select features for modeling
X = combined_data[["encoded_venue", "encoded_match_type"]]
y = combined_data["winner"]

# Build a predictive model (Random Forest Classifier)
model = RandomForestClassifier(random_state=42)
model.fit(X, y)


# Make predictions for new data
def predict_match(
    teamA_encoded_venue,
    teamA_encoded_match_type,
):
    new_data = [[teamA_encoded_venue, teamA_encoded_match_type]]
    prediction = model.predict(new_data)
    return prediction[0]
