import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np

# Load the dataset
data = pd.read_csv("dataNepal.csv")

# Encode the categorical columns
label_encoder = LabelEncoder()
data["Opponent"] = label_encoder.fit_transform(data["Opponent"])
data["Location"] = label_encoder.fit_transform(data["Location"])
data["Toss Winner"] = label_encoder.fit_transform(data["Toss Winner"])

# Split the dataset into features and target
X = data.drop(["Result", "Score"], axis=1)
y = data["Result"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Train the logistic regression model with increased max_iter
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)


# Function to calculate opponent's strength based on ICC rank
def get_opponent_strength(opponent):
    opponent_rank = label_encoder.transform([opponent])[0]
    opponent_column = data.iloc[:, opponent_rank]

    # Exclude non-numeric values from the calculation
    opponent_strength = opponent_column.loc[
        opponent_column.apply(lambda x: pd.to_numeric(x, errors="coerce")).notna()
    ].mean()

    return opponent_strength


# Function to calculate location factor based on Nepal's performance in that location
def get_location_factor(location):
    location_data = data[data["Location"] == label_encoder.transform([location])[0]]
    location_factor = location_data["Total Runs"].mean() / data["Total Runs"].mean()
    return location_factor


# Function to make prediction and estimate total runs range
def make_prediction(opponent, location, toss_winner):
    opponent_strength = get_opponent_strength(opponent)
    location_factor = get_location_factor(location)
    toss_winner_factor = 1.0  # Default toss winner factor

    # Check for NaN values and replace with 0
    if np.isnan(opponent_strength):
        opponent_strength = 0.0
    if np.isnan(location_factor):
        location_factor = 0.0

    # Make predictions on new data
    new_data = pd.DataFrame(
        {
            "Opponent": label_encoder.transform([opponent])[0],
            "Location": label_encoder.transform([location])[0],
            "Toss Winner": label_encoder.transform([toss_winner])[0],
            "Total Runs": [data["Total Runs"].mean()],  # Add a placeholder value
            "Opponent_Strength": [opponent_strength],
            "Location_Factor": [location_factor],
            "Toss_Winner_Factor": [toss_winner_factor],
        }
    )
    new_data = imputer.transform(new_data)
    prediction = clf.predict(new_data)

    # Calculate the estimated range using the added parameters
    total_runs_mean = int(data["Total Runs"].mean())
    total_runs_std = int(data["Total Runs"].std())

    # Calculate the opponent factor
    opponent_factor = opponent_strength * location_factor

    # Calculate the range multiplier based on toss winner
    if toss_winner == "Nepal":
        range_multiplier = 0.2
    else:
        range_multiplier = 0.1

    # Calculate the low and high range of estimated total runs
    range_low = int(
        total_runs_mean - total_runs_std * range_multiplier + opponent_factor
    )
    range_high = int(
        total_runs_mean + total_runs_std * range_multiplier + opponent_factor
    )

    # Make predictions based on the range
    if range_low <= 0:
        prediction = "Defeat"
    elif range_high >= 500:
        prediction = "Victory"
    else:
        prediction = "Neutral"

    return f"Result: {prediction}, Estimated range of total runs: {range_low} - {range_high}"
