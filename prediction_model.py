import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import joblib

class MatchPredictionModel:
    def __init__(self, data_folder, historical_data_folder, model_file):
        self.data_folder = data_folder
        self.historical_data_folder = historical_data_folder
        self.model_file = model_file
        self.unique_venues = []  # Store unique venues as an attribute
        self.unique_pitch_conditions = []  # Store unique pitch conditions as an attribute

    def preprocess_data(self):
        match_data = []

        for filename in os.listdir(self.data_folder):
            if filename.endswith(".json"):
                with open(os.path.join(self.data_folder, filename)) as file:
                    match = json.load(file)
                    match_data.append(match)

        return match_data

    def collect_historical_data(self):
        historical_match_data = []

        for filename in os.listdir(self.historical_data_folder):
            if filename.endswith(".json"):
                with open(os.path.join(self.historical_data_folder, filename)) as file:
                    match = json.load(file)
                    historical_match_data.append(match)

        return historical_match_data

    def extract_features(self, match_data):
        features = []
        labels = []

        venues = set()
        pitch_conditions = set()

        for match in match_data:
            if "info" in match and "outcome" in match["info"]:
                outcome = match["info"]["outcome"]
                if "winner" in outcome:
                    winner = outcome["winner"]
                    if winner in match["info"]["teams"]:
                        label = 0 if match["info"]["teams"].index(winner) == 0 else 1
                        labels.append(label)

                        venue = match["info"]["venue"]
                        overs = match["info"]["overs"]
                        pitch_condition = match["info"].get("pitch_condition", "Unknown")
                        opposition = (
                            match["info"]["teams"][1]
                            if match["info"]["teams"].index(winner) == 0
                            else match["info"]["teams"][0]
                        )

                        venues.add(venue)
                        pitch_conditions.add(pitch_condition)

                        features.append([opposition, overs, venue, pitch_condition])

        unique_venues = list(venues)
        unique_pitch_conditions = list(pitch_conditions)
        self.unique_venues = list(venues)
        self.unique_pitch_conditions = list(pitch_conditions)

        # Create one-hot encoders for venues and pitch conditions
        venue_encoder = OneHotEncoder(sparse=False)
        pitch_condition_encoder = OneHotEncoder(sparse=False)

        venue_encoded = venue_encoder.fit_transform(np.array(unique_venues).reshape(-1, 1))
        pitch_condition_encoded = pitch_condition_encoder.fit_transform(np.array(unique_pitch_conditions).reshape(-1, 1))

        venue_mapping = {venue: feature.tolist() for venue, feature in zip(unique_venues, venue_encoded)}
        pitch_condition_mapping = {condition: feature.tolist() for condition, feature in zip(unique_pitch_conditions, pitch_condition_encoded)}

        # Convert venue strings and pitch conditions to their corresponding one-hot encoded features in the dataset
        for i, feature_set in enumerate(features):
            venue = feature_set[2]
            pitch_condition = feature_set[3]
            venue_feature = venue_mapping.get(venue, [0] * len(unique_venues))
            pitch_condition_feature = pitch_condition_mapping.get(pitch_condition, [0] * len(unique_pitch_conditions))
            features[i] = [feature_set[0], feature_set[1]] + venue_feature + pitch_condition_feature

        return features, labels

    def train_model(self, features, labels):
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (Combined Features):", accuracy)

        joblib.dump(model, self.model_file)

    def load_trained_model(self):
        try:
            model = joblib.load(self.model_file)
            return model
        except FileNotFoundError:
            raise FileNotFoundError(f"Trained model file not found at: {self.model_file}")

    def predict(self, new_data):
        model = self.load_trained_model()
        predictions = model.predict(new_data)
        return predictions

    def run(self):
        match_data = self.preprocess_data()
        historical_match_data = self.collect_historical_data()
        all_match_data = match_data + historical_match_data
        features, labels = self.extract_features(all_match_data)
        self.train_model(features, labels)

if __name__ == "__main__":
    data_folder = "nepal_male_json"
    historical_data_folder = "nepal_male_json"
    model_file = "trained_model.pkl"
    prediction_model = MatchPredictionModel(data_folder, historical_data_folder, model_file)
    prediction_model.run()
