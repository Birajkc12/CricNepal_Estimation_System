import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder


class MatchPredictionModel:
    def __init__(self, data_folder):
        self.data_folder = data_folder

    def preprocess_data(self):
        match_data = []

        for filename in os.listdir(self.data_folder):
            if filename.endswith(".json"):
                with open(os.path.join(self.data_folder, filename)) as file:
                    match = json.load(file)
                    match_data.append(match)

        return match_data

    def extract_features(self, match_data):
        features = []
        labels = []

        venues = set()  # To store unique venues

        for match in match_data:
            if "info" in match and "outcome" in match["info"]:
                outcome = match["info"]["outcome"]
                if "winner" in outcome:
                    winner = outcome["winner"]
                    if winner in match["info"]["teams"]:
                        label = 0 if match["info"]["teams"].index(winner) == 0 else 1
                        labels.append(label)

                        team1_players = match["info"]["players"][
                            match["info"]["teams"][0]
                        ]
                        team2_players = match["info"]["players"][
                            match["info"]["teams"][1]
                        ]
                        venue = match["info"]["venue"]
                        overs = match["info"]["overs"]

                        # Add the venue to the set of unique venues
                        venues.add(venue)

                        features.append(
                            [len(team1_players), len(team2_players), overs, venue]
                        )

        # Convert the set of venues into a list
        unique_venues = list(venues)

        # Create the one-hot encoder
        encoder = OneHotEncoder(sparse=False)

        # Fit and transform the venues into one-hot encoded features
        venue_encoded = encoder.fit_transform(np.array(unique_venues).reshape(-1, 1))

        # Create a mapping of venues to their corresponding one-hot encoded feature
        venue_mapping = {
            venue: feature for venue, feature in zip(unique_venues, venue_encoded)
        }

        # Convert venue strings to their corresponding one-hot encoded features in the dataset
        for i, feature_set in enumerate(features):
            venue = feature_set[3]
            features[i][3] = venue_mapping[venue]

        return features, labels

    def train_model(self, features, labels):
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

        # Split the concatenated features back into numerical and venue features
        X_train_numerical = [x[:3] for x in X_train]
        X_train_venue = [x[3] for x in X_train]

        X_test_numerical = [x[:3] for x in X_test]
        X_test_venue = [x[3] for x in X_test]

        # Create the RandomForestClassifier
        model = RandomForestClassifier()

        # Fit the model using the numerical features
        model.fit(X_train_numerical, y_train)

        # Predict using the numerical features
        y_pred = model.predict(X_test_numerical)

        # Calculate accuracy using the numerical features
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (Numerical Only):", accuracy)

        # Now, let's predict using both numerical and venue features
        y_pred_combined = model.predict(X_test_numerical)
        accuracy_combined = accuracy_score(y_test, y_pred_combined)
        print("Accuracy (Combined Numerical and Venue):", accuracy_combined)

    def run(self):
        match_data = self.preprocess_data()
        features, labels = self.extract_features(match_data)
        self.train_model(features, labels)


if __name__ == "__main__":
    data_folder = "nepal_male_json"
    prediction_model = MatchPredictionModel(data_folder)
    prediction_model.run()
