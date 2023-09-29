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

# Ensure that 'Unknown' label is replaced with 'Neutral' in y
data["Result"].replace("Unknown", "Neutral", inplace=True)

# Split the dataset into features and target
X = data.drop(["Result", "Score"], axis=1)
y = data["Result"].astype("category")
y = y.cat.set_categories(["Victory", "Defeat", "Neutral"])

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
    try:
        opponent_rank = label_encoder.transform([opponent])[0]
    except ValueError:
        # Handle previously unseen opponents by setting rank to -1
        opponent_rank = -1

    if opponent_rank != -1:
        opponent_column = data.iloc[:, opponent_rank]

        # Exclude non-numeric values from the calculation
        opponent_strength = opponent_column.loc[
            opponent_column.apply(lambda x: pd.to_numeric(x, errors="coerce")).notna()
        ].mean()
    else:
        # Handle unseen opponents gracefully
        opponent_strength = 0.0

    return opponent_strength


# Function to calculate location factor based on Nepal's performance in that location
def get_location_factor(location):
    if location not in label_encoder.classes_:
        # Handle previously unseen location, e.g., assign it as "Unknown"
        return 1.0  # Neutral factor when location is unknown

    location_encoded = label_encoder.transform([location])[0]

    location_data = data[data["Location"] == location_encoded]

    # Handle the case when location_data is empty (unseen location)
    if location_data.empty:
        # You can assign a default value or take appropriate action here.
        # For example, assigning a neutral factor:
        return 1.0  # Neutral factor when location is unknown

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

    # Handle previously unseen labels in y
    if y.cat.categories.isin(["Victory", "Defeat", "Neutral"]).all():
        # All labels are known, proceed with predictions
        pass
    else:
        # Set previously unseen labels to 'Neutral'
        y.cat.add_categories(["Neutral"], inplace=True)
        y.fillna("Neutral", inplace=True)

    # Make predictions on new data
    new_data = pd.DataFrame(
        {
            "Opponent": [opponent_strength],
            "Location": [location_factor],
            "Toss Winner": [toss_winner_factor],
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
