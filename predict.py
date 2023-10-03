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

# ICC rankings data with points and ratings
icc_rankings = {
   "India": {"rank": 1, "points": 15589, "rating": 264},
    "England": {"rank": 2, "points": 11133, "rating": 259},
    "New Zealand": {"rank": 3, "points": 13534, "rating": 255},
    "Pakistan": {"rank": 4, "points": 12719, "rating": 254},
    "Australia": {"rank": 5, "points": 7984, "rating": 250},
    "South Africa": {"rank": 6, "points": 8679, "rating": 248},
    "West Indies": {"rank": 7, "points": 9463, "rating": 243},
    "Sri Lanka": {"rank": 8, "points": 8774, "rating": 237},
    "Bangladesh": {"rank": 9, "points": 9192, "rating": 224},
    "Afghanistan": {"rank": 10, "points": 6260, "rating": 216},
    "Zimbabwe": {"rank": 11, "points": 7087, "rating": 197},
    "Ireland": {"rank": 12, "points": 8487, "rating": 193},
    "Namibia": {"rank": 13, "points": 3228, "rating": 190},
    "Scotland": {"rank": 14, "points": 3412, "rating": 190},
    "Netherlands": {"rank": 15, "points": 3445, "rating": 181},
    "Nepal": {"rank": 16, "points": 2679, "rating": 179},
    "UAE": {"rank": 17, "points": 4591, "rating": 177},
    "Papua New Guinea": {"rank": 18, "points": 3173, "rating": 144},
    "Canada": {"rank": 19, "points": 1871, "rating": 144},
    "Oman": {"rank": 20, "points": 2723, "rating": 143},
    "Hong Kong": {"rank": 21, "points": 2530, "rating": 133},
    "United States": {"rank": 22, "points": 1183, "rating": 131},
    "Uganda": {"rank": 23, "points": 7043, "rating": 130},
    "Jersey": {"rank": 24, "points": 2427, "rating": 128},
    "Malaysia": {"rank": 25, "points": 4537, "rating": 126},
    "Qatar": {"rank": 26, "points": 1568, "rating": 121},
    "Kuwait": {"rank": 27, "points": 2388, "rating": 119},
    "Bahrain": {"rank": 28, "points": 3518, "rating": 110},
    "Kenya": {"rank": 29, "points": 3137, "rating": 105},
    "Italy": {"rank": 30, "points": 1712, "rating": 101},
    "Tanzania": {"rank": 31, "points": 4286, "rating": 100},
    "Bermuda": {"rank": 32, "points": 862, "rating": 96},
    "Spain": {"rank": 33, "points": 1649, "rating": 92},
    "Germany": {"rank": 34, "points": 3022, "rating": 92},
    "Saudi Arabia": {"rank": 35, "points": 1313, "rating": 88},
    "Singapore": {"rank": 36, "points": 1910, "rating": 80},
    "Guernsey": {"rank": 37, "points": 1212, "rating": 76},
    "Cayman Islands": {"rank": 38, "points": 521, "rating": 74},
    "Portugal": {"rank": 39, "points": 1167, "rating": 73},
    "Denmark": {"rank": 40, "points": 1622, "rating": 71},
    "Belgium": {"rank": 41, "points": 1237, "rating": 69},
    "Nigeria": {"rank": 42, "points": 1026, "rating": 68},
    "Isle of Man": {"rank": 43, "points": 949, "rating": 63},
    "Austria": {"rank": 44, "points": 1682, "rating": 58},
    "Norway": {"rank": 45, "points": 852, "rating": 57},
    "Vanuatu": {"rank": 46, "points": 846, "rating": 56},
    "Botswana": {"rank": 47, "points": 1300, "rating": 54},
    "Finland": {"rank": 48, "points": 953, "rating": 53},
    "Switzerland": {"rank": 49, "points": 835, "rating": 52},
    "Japan": {"rank": 50, "points": 726, "rating": 52},
    "Malawi": {"rank": 51, "points": 682, "rating": 49},
    "Czech Republic": {"rank": 52, "points": 1101, "rating": 46},
    "France": {"rank": 53, "points": 730, "rating": 46},
    "Sweden": {"rank": 54, "points": 759, "rating": 42},
    "Romania": {"rank": 55, "points": 1149, "rating": 41},
    "Cook Islands": {"rank": 56, "points": 245, "rating": 41},
    "Indonesia": {"rank": 57, "points": 372, "rating": 37},
    "Mozambique": {"rank": 58, "points": 759, "rating": 36},
    "Sierra Leone": {"rank": 59, "points": 421, "rating": 35},
    # Add more teams with their data as needed
}


# Function to calculate opponent's strength based on ICC rank, points, and ratings
def get_opponent_strength(opponent):
    opponent_data = icc_rankings.get(opponent)

    if opponent_data:
        opponent_rank = opponent_data["rank"]
        opponent_points = opponent_data["points"]
        opponent_rating = opponent_data["rating"]

        # Define a mapping function based on your criteria for strength calculation
        # For example, you can consider the inverse of the rank
        opponent_strength = (
            0.5 * (1 / opponent_rank)
            + 0.3 * (opponent_points / 10000)
            + 0.2 * (opponent_rating / 300)
        )
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



# Function to calculate win percentage between Nepal and the opponent
def calculate_win_percentage(opponent, location):
    # Replace this with your actual win percentage calculation logic
    # For demonstration purposes, I'll provide a simple example using hypothetical data.

    # Hypothetical win data (you should replace this with your actual data)
    win_data = {
        "India": {
            "Home": 0.75,  # Nepal's win percentage against India at Home
            "Away": 0.25,  # Nepal's win percentage against India Away
        },
        "England": {
            "Home": 0.65,  # Nepal's win percentage against England at Home
            "Away": 0.35,  # Nepal's win percentage against England Away
        },
        # Add more opponents and win percentages as needed
    }

    # Check if the opponent and location are in the win_data
    if opponent in win_data and location in win_data[opponent]:
        win_percentage = win_data[opponent][location]
    else:
        # Handle cases where there is no historical data available
        # You can assign a default win percentage or use other strategies.
        win_percentage = 0.5  # Default win percentage (50%)

    return win_percentage

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

    # Calculate the win percentage between Nepal and the opponent
    # You can replace this with your actual win percentage calculation
    win_percentage = calculate_win_percentage(opponent, location)


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
    return f"Result: {prediction}, Estimated range of total runs: {range_low} - {range_high}"
