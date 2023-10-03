from flask import (
    Flask,
    render_template,
    request,
    session,
    redirect,
    send_from_directory,
    jsonify,
)
import requests
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Import DecisionTreeRegressor from sklearn.tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from playerprofile import get_player_stats
from werkzeug.utils import secure_filename
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
import pickle
import csv
import joblib
import json
import prediction_module
import numpy as np  # Import NumPy for data manipulation
from predict import make_prediction
from prediction_model import MatchPredictionModel

from flask_cors import CORS

app = Flask(__name__)
app.secret_key = "64131316905e8dea79bc77eda222e351"
CORS(app)  # Enable CORS for your Flask app


# Load the dataset
data = pd.read_csv("player_performance.csv")


# Calculate additional metrics
data["Image"] = data["Player"].str.replace(" ", "_") + ".jpg"
data["Above_Avg_Batting"] = data["Batting_Average"] > data["Batting_Average"].mean()
data["Above_Avg_Bowling"] = data["Bowling_Average"] > data["Bowling_Average"].mean()


# Function to calculate batting average
def calculate_batting_average(row):
    if row["Innings"] > 0:
        return row["Runs"] / row["Innings"]
    else:
        return 0


# Function to calculate bowling average
def calculate_bowling_average(row):
    if row["Wickets"] > 0:
        return row["Runs"] / row["Wickets"]
    else:
        return 0


# Function to calculate strike rate
def calculate_strike_rate(row):
    if row["Innings"] > 0 and row["Balls_Faced"] > 0:
        strike_rate = (row["Runs"] / row["Balls_Faced"]) * 100
        return strike_rate
    else:
        return 0.0


# Function to calculate economy rate
def calculate_economy_rate(row):
    if row["Wickets"] > 0:
        return (row["Runs"] / row["Wickets"]) * 6
    else:
        return 0


# Calculate additional metrics
data["Batting_Average"] = data.apply(calculate_batting_average, axis=1)
data["Bowling_Average"] = data.apply(calculate_bowling_average, axis=1)
data["Strike_Rate"] = data.apply(calculate_strike_rate, axis=1)
data["Economy_Rate"] = data.apply(calculate_economy_rate, axis=1)

# Drop samples with missing values
data.dropna(inplace=True)

# Prepare the data for training
X = data[
    [
        "Matches",
        "Innings",
        "Runs",
        "Wickets",
        "Batting_Average",
        "Bowling_Average",
        "Strike_Rate",
        "Economy_Rate",
    ]
]
y_batting = data["Above_Avg_Batting"]
y_bowling = data["Above_Avg_Bowling"]

# Split the data into training and testing sets
X_train_batting, X_test_batting, y_train_batting, y_test_batting = train_test_split(
    X, y_batting, test_size=0.2, random_state=42
)
X_train_bowling, X_test_bowling, y_train_bowling, y_test_bowling = train_test_split(
    X, y_bowling, test_size=0.2, random_state=42
)

# Build decision tree classifiers
clf_batting = DecisionTreeClassifier()
clf_bowling = DecisionTreeClassifier()

# Train the classifiers
clf_batting.fit(X_train_batting, y_train_batting)
clf_bowling.fit(X_train_bowling, y_train_bowling)


# Route for the home page
@app.route("/")
def home():
    return render_template("index.html")


# Route for the analysis page
import numpy as np


@app.route("/analysis", methods=["GET", "POST"])
def analysis():
    search_query = request.form.get("search_query")
    if search_query:
        filtered_data = data[data["Player"].str.contains(search_query, case=False)]
    else:
        filtered_data = data

    batting_predictions = (
        clf_batting.predict(filtered_data[X.columns]).astype(bool).ravel()
    )
    bowling_predictions = (
        clf_bowling.predict(filtered_data[X.columns]).astype(bool).ravel()
    )

    above_average_batting = filtered_data.loc[
        np.logical_or(
            batting_predictions, filtered_data["Above_Avg_Batting"].astype(bool)
        )
    ]
    above_average_bowling = filtered_data.loc[
        np.logical_or(
            bowling_predictions, filtered_data["Above_Avg_Bowling"].astype(bool)
        )
    ]

    batting_chart_data = {
        "labels": above_average_batting["Player"].tolist(),
        "data": above_average_batting["Batting_Average"].tolist(),
    }

    bowling_chart_data = {
        "labels": above_average_bowling["Player"].tolist(),
        "data": above_average_bowling["Bowling_Average"].tolist(),
    }

    return render_template(
        "analysis.html",
        players=filtered_data.to_dict("records"),
        above_average_batting=above_average_batting.to_dict("records"),
        above_average_bowling=above_average_bowling.to_dict("records"),
        batting_chart_data=batting_chart_data,
        bowling_chart_data=bowling_chart_data,
    )


@app.route("/player/<player_name>")
def player(player_name):
    # Retrieve the player's stats based on the player_name parameter
    player = get_player_stats(player_name)

    # Render the player.html template with the player's stats
    return render_template("player.html", player=player)


# Function to get all player names from the dataset
def get_all_players(data):
    return data["Player"].values.tolist()


# Function to sanitize the player's name for generating the image filename
def sanitize_player_name(player_name):
    # Remove special characters and replace spaces with underscores
    sanitized_name = "".join(c for c in player_name if c.isalnum() or c.isspace())
    sanitized_name = sanitized_name.replace(" ", "_")
    return sanitized_name


# Route for the admin panel
@app.route("/admin", methods=["GET", "POST"])
def admin_panel():
    global data  # Add this line to access the global 'data' variable

    # Check if the admin is logged in
    if session.get("admin"):
        if request.method == "POST":
            # Retrieve the player information from the form
            player_name = request.form["player_name"]
            matches = int(request.form["matches"])
            innings = int(request.form["innings"])
            runs = int(request.form["runs"])
            wickets = int(request.form["wickets"])
            balls_faced = int(request.form["balls_faced"])
            player_image = request.files["player_image"]

            # Check if a player image was uploaded
            if "player_image" in request.files:
                player_image = request.files["player_image"]
                if player_image.filename != "":
                    filename = secure_filename(player_image.filename)
                    player_name = sanitize_player_name(
                        player_name
                    )  # Sanitize the player's name
                    filename = (
                        player_name + os.path.splitext(filename)[1]
                    )  # Use sanitized player's name as the filename
                    player_image.save(os.path.join("static/player_images", filename))

            if player_name not in data["Player"].values:
                # Perform validation and save the player to the dataset
                # Add the player information to the 'data' DataFrame
                new_player = {
                    "Player": player_name,
                    "Matches": matches,
                    "Innings": innings,
                    "Runs": runs,
                    "Wickets": wickets,
                    "Balls_Faced": balls_faced,
                }
                data = data.append(new_player, ignore_index=True)
            else:
                # Update the player information in the 'data' DataFrame
                player_index = data.index[data["Player"] == player_name].tolist()[0]

                data.at[player_index, "Matches"] += matches
                data.at[player_index, "Innings"] += innings
                data.at[player_index, "Runs"] += runs
                data.at[player_index, "Wickets"] += wickets
                data.at[player_index, "Balls_Faced"] += balls_faced

            # Save the player image
            if player_image:
                filename = secure_filename(player_image.filename)
                player_image.save(os.path.join("static/player_images", filename))

            # Recalculate the batting average, bowling average, strike rate, and economy rate
            data["Batting_Average"] = data.apply(calculate_batting_average, axis=1)
            data["Bowling_Average"] = data.apply(calculate_bowling_average, axis=1)
            data["Strike_Rate"] = data.apply(calculate_strike_rate, axis=1)
            data["Economy_Rate"] = data.apply(calculate_economy_rate, axis=1)

            # Save the updated dataset to the CSV file
            data.to_csv("player_performance.csv", index=False)

            return redirect("/admin")

        players = get_all_players(data)  # Get all player names from the dataset

        return render_template("admin_panel.html", players=players)
    else:
        # Redirect to the admin login page if not logged in
        return redirect("/admin/login")


# Route for the admin login
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # Add your authentication logic here
        # For example, you can check if the username and password match a predefined admin account

        if username == "admin" and password == "password":
            session["admin"] = True  # Set the admin flag in the session
            return redirect("/admin")

        return render_template("admin_login.html", error="Invalid username or password")

    return render_template("admin_login.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file uploaded", 400

    player_image = request.files["file"]
    if player_image.filename == "":
        return "Invalid file", 400

    filename, file_extension = os.path.splitext(player_image.filename)
    file_extension = file_extension.lower()
    filename = secure_filename(filename + file_extension)
    player_image.save(os.path.join("static/player_images", filename))

    return "File uploaded successfully"


# Load the initial dataset from the CSV file
data = pd.read_csv("player_performance.csv")


@app.route("/admin/delete", methods=["POST"])
def delete_player():
    global data  # Use the global 'data' variable

    player_name = request.form["player_name"]

    # Delete the player from the 'data' DataFrame
    data = data[data["Player"] != player_name]

    # Save the updated dataset to the CSV file
    data.to_csv("player_performance.csv", index=False)

    return redirect("/admin")


@app.route("/prediction")
def prediction():
    return render_template("prediction.html")


# Load the dataset
data = pd.read_csv("player_performance.csv")

# Handle missing values
data.fillna(0, inplace=True)  # Replace NaN values with 0


# Route for the prediction
@app.route("/predict", methods=["POST"])
def predict():
    opponent = request.form["opponent"]
    location = request.form["location"]
    toss_winner = request.form["toss_winner"]

    # Use the make_prediction function from predict.py
    prediction = make_prediction(opponent, location, toss_winner)

    return jsonify({"prediction": prediction})

    # return render_template("prediction.html", prediction=prediction, icc_rankings=icc_rankings)


# Load player data from CSV file
def load_player_data(filename):
    players = []
    with open(filename, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            players.append(row)
    return players


# Route for the player comparison page
@app.route("/compare", methods=["GET", "POST"])
def compare_players():
    if request.method == "POST":
        player1_name = request.form["player1"]
        player2_name = request.form["player2"]

        players = load_player_data("player_performance.csv")
        selected_player1 = next(
            (player for player in players if player["Player"] == player1_name), None
        )
        selected_player2 = next(
            (player for player in players if player["Player"] == player2_name), None
        )

        if selected_player1 is None or selected_player2 is None:
            error_message = "One or both players are not found."
            return render_template("error.html", error_message=error_message)

        return render_template(
            "compare.html",
            players=players,
            player1=selected_player1,
            player2=selected_player2,
        )

    # If the request method is GET, display the comparison form
    return render_template("compare.html", players=players)


# Load player data from CSV file
def load_player_data(file_path):
    player_data = []
    with open(file_path, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            player_data.append(row)
    return player_data


# Initialize player data
players = load_player_data("player_performance.csv")


# Route for the analysis page
@app.route("/analyze", methods=["POST"])
def analyze_performance():
    selected_player = request.form["player"]
    pitch_type = request.form["pitch_type"]
    weather = request.form["weather"]
    algorithm = request.form["algorithm"]

    # Find the selected player's data from the players list
    player1 = next(
        (player for player in players if player["Player"] == selected_player), None
    )

    if player1 is None:
        error_message = "Selected player not found in the dataset."
        return render_template("error.html", error_message=error_message)

    # Pass player1 data, pitch_type, and weather to the analyze_player_performance function
    prediction = analyze_player_performance(player1, pitch_type, weather, algorithm)

    return render_template(
        "analysis_result.html",
        prediction=prediction,
        player_image=player1["Image"],
    )


def predict_linear_regression(player, pitch_type, weather):
    # Convert the list of dictionaries to a DataFrame
    df_players = pd.DataFrame(players)

    # Use batting and bowling averages as features for prediction
    X = df_players[["Batting_Average", "Bowling_Average"]]
    y = df_players["Runs"]

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict runs for the selected player
    predicted_runs = model.predict(
        [[float(player["Batting_Average"]), float(player["Bowling_Average"])]]
    )

    return {"Predicted_Runs": predicted_runs[0]}


def predict_decision_tree(player, pitch_type, weather):
    # Convert the list of dictionaries to a DataFrame
    df_players = pd.DataFrame(players)

    # Use batting and bowling averages as features for prediction
    X = df_players[["Batting_Average", "Bowling_Average"]]
    y = df_players["Runs"]

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create a decision tree regressor model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predict runs for the selected player
    predicted_runs = model.predict(
        [[float(player["Batting_Average"]), float(player["Bowling_Average"])]]
    )

    return {"Predicted_Runs": predicted_runs[0]}


# ---------------important one--------------------------
def analyze_player_performance(player, pitch_type, weather, algorithm):
    # ... (your analysis code)

    predicted_performance = {
        "Player": player["Player"],
        "Predicted_Batting_Average": float(player["Batting_Average"])
        + get_batting_adjustment(pitch_type, weather),
        "Predicted_Bowling_Average": float(player["Bowling_Average"])
        + get_bowling_adjustment(pitch_type, weather),
        "Pitch_Type": pitch_type,
        "Weather": weather,
        "PlayerImage": player["Image"],  # Include the player's image filename
        # ... (other predictions)
    }
    return predicted_performance


# Define functions to get adjustment values based on pitch type and weather
def get_batting_adjustment(pitch_type, weather):
    if pitch_type == "Green" and weather == "Rainy":
        return -5.0  # Adjust batting average for challenging conditions
    elif pitch_type == "Dry" and weather == "Sunny":
        return 2.0  # Adjust batting average for favorable conditions
    else:
        return 0.0  # No adjustment for other conditions


def get_bowling_adjustment(pitch_type, weather):
    if pitch_type == "Green" and weather == "Rainy":
        return 3.0  # Adjust bowling average for challenging conditions
    elif pitch_type == "Dry" and weather == "Sunny":
        return -2.0  # Adjust bowling average for favorable conditions
    else:
        return 0.0  # No adjustment for other conditions


# ---------------important one--------------------------


# # Initialize the prediction model
# data_folder = "nepal_male_json"
# prediction_model = MatchPredictionModel(data_folder)
# prediction_model.run()


# # Route to handle prediction
# @app.route("/make_prediction", methods=["GET", "POST"])
# def make_prediction():
#     if request.method == "POST":
#         # Retrieve input features from the frontend form
#         team1 = request.form.get("team1")
#         team2 = request.form.get("team2")
#         venue = request.form.get("venue")
#         overs = request.form.get("overs")

#         # Check if the "overs" input is not empty before converting to int
#         if overs:
#             overs = int(overs)

#             # Use the prediction model to predict the winner
#             new_match_features = [len(team1), len(team2), overs, venue]
#             predicted_winner = prediction_model.predict_winner(new_match_features)

#             return render_template(
#                 "prediction_result.html", predicted_winner=predicted_winner
#             )

#     return render_template("sample_prediction.html")


# Route to display sample data
@app.route("/sample_data")
def sample_data():
    # Load match data from the dataset
    match_data = prediction_model.preprocess_data()

    # Sort the match data by date in descending order to show the latest matches first  
    match_data.sort(key=lambda match: match["info"]["dates"][0], reverse=True)

    # Pagination settings
    page = request.args.get("page", 1, type=int)
    matches_per_page = 5
    start_idx = (page - 1) * matches_per_page
    end_idx = start_idx + matches_per_page

    paginated_matches = match_data[start_idx:end_idx]

    total_pages = (len(match_data) // matches_per_page) + 1

    return render_template(
        "sample_data.html",
        paginated_matches=paginated_matches,
        page=page,
        total_pages=total_pages,
    )


@app.route("/match_details/<string:match_id>")
def match_details(match_id):
    # Assuming match JSON files are stored in the "nepal_male_json" directory
    match_path = os.path.join(app.root_path, "nepal_male_json", f"{match_id}.json")

    # Load and parse the match JSON content
    with open(match_path, "r") as file:
        match_data = json.load(file)

    return render_template("match_details.html", match_data=match_data)




# Load the trained model when the Flask app starts
model_file = "trained_model.pkl"  # Replace with the actual path to your trained model file
model = joblib.load(model_file)

# Load venue and pitch condition mappings (ensure these mappings are correctly defined)
venue_mapping = {
    "Nepal": [0, 1, 0],  # Example one-hot encoding for Nepal
    "UAE": [1, 0, 0],    # Example one-hot encoding for UAE
    # Add more venues and their encodings as needed
}

pitch_condition_mapping = {
    "Dry": [0, 1],      # Example one-hot encoding for Dry pitch condition
    "Wet": [1, 0],      # Example one-hot encoding for Wet pitch condition
    "Unknown": [0, 0],  # Example one-hot encoding for Unknown pitch condition
    # Add more pitch conditions and their encodings as needed
}

# Initialize demo ICC rankings for opposition team strength
icc_rankings = {
    "India": 1,
    "Australia": 2,
    "England": 3,
    "Pakistan": 4,
    "South Africa": 5,
    # Add more teams and rankings as needed
}


# # Render the predictions.html page
# @app.route('/predictions', methods=['GET', 'POST'])
# def make_predictions():

#     if request.method == 'POST':
#         # Retrieve data from the form
#         overs = float(request.form['overs'])
#         venue = request.form['venue']
#         opposition = request.form['opposition']
#         pitch_condition = request.form['pitch_condition']
#         bat_or_bowl = request.form['bat_or_bowl']

#         # Calculate opposition team strength based on ICC rankings
#         opposition_strength = icc_rankings.get(opposition, 0)

#         # Ensure 'unique_venues' is available here (assuming 'prediction_model' is an instance of MatchPredictionModel)
#         unique_venues = []  # Replace with the actual variable name

#         # Prepare the input data for prediction
#         venue_feature = venue_mapping.get(venue, [0, 0, 0])  # Use venue_mapping
#         pitch_condition_feature = pitch_condition_mapping.get(pitch_condition, [0, 0])  # Use pitch_condition_mapping

#         # Define missing features with default values (you may adjust these)
#         missing_features = [0.0, 0.0]  # Default values for missing features

#         # Combine the features into an input array
#         input_data = np.array([overs] + venue_feature + pitch_condition_feature + [opposition_strength, 0.0])


#         # Make predictions using the model
#         prediction = model.predict([input_data])[0]


#         if bat_or_bowl == 'bat':
#          # Round the prediction value to the nearest whole number
#          rounded_prediction = round(prediction)
#          # Define a range (you can adjust this range as needed)
#          prediction_range = f"{rounded_prediction - 10} to {rounded_prediction + 10} runs"
#          prediction_label = 'runs'
#     else:
#         # Similar logic for wickets
#         rounded_prediction = round(prediction)
#         prediction_range = f"{rounded_prediction - 2} to {rounded_prediction + 2} wickets"
#         prediction_label = 'wickets'

        # return render_template('predictions.html', prediction=prediction_range, bat_or_bowl=bat_or_bowl)

    # Handle GET request (initial form display)
    # return render_template('predictions.html', prediction=None)

# # Split the data into training and testing sets for the batting random forest classifier
# X_train_rf_batting, X_test_rf_batting, y_train_rf_batting, y_test_rf_batting = train_test_split(
#     X, y_batting, test_size=0.2, random_state=42
# )

# # Build and train the batting random forest classifier
# rf_classifier_batting = RandomForestClassifier(random_state=42)
# rf_classifier_batting.fit(X_train_rf_batting, y_train_rf_batting)

# # Split the data into training and testing sets for the bowling random forest classifier
# X_train_rf_bowling, X_test_rf_bowling, y_train_rf_bowling, y_test_rf_bowling = train_test_split(
#     X, y_bowling, test_size=0.2, random_state=42
# )

# # Build and train the bowling random forest classifier
# rf_classifier_bowling = RandomForestClassifier(random_state=42)
# rf_classifier_bowling.fit(X_train_rf_bowling, y_train_rf_bowling)

# # # Train a single Random Forest Classifier
# # rf_classifier = RandomForestClassifier(random_state=42)
# # # Train the random forest classifier using the batting dataset
# # rf_classifier.fit(X_train_rf_batting, y_train_rf_batting)

# # # Evaluate Random Forest for batting
# # rf_accuracy_batting = rf_classifier_batting.score(X_test_rf_batting, y_test_rf_batting)
# # # Evaluate Random Forest for bowling
# # rf_accuracy_bowling = rf_classifier_bowling.score(X_test_rf_bowling, y_test_rf_bowling)

# from sklearn.datasets import load_iris

# # Load the Iris dataset as an example
# iris = load_iris()
# X = iris.data  # Feature matrix
# y = iris.target  # Target variable

# # Split your data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # Create a logistic regression classifier
# lr_classifier = LogisticRegression(random_state=42)

# # Train the logistic regression classifier
# lr_classifier.fit(X_train, y_train)  # You should replace X_train and y_train with your actual training data

# # Evaluate Logistic Regression
# lr_accuracy = lr_classifier.score(X_test, y_test)  # Calculate accuracy on the test data

# # Evaluate Random Forest
# rf_accuracy = rf_classifier.score(X_test, y_test)

# # Define a route to display the comparison results
# @app.route('/comparison')
# def show_comparison():
#     global lr_accuracy, rf_accuracy, comparison_chart

#     # Ensure that both algorithms have been run
#     if lr_accuracy is not None and rf_accuracy is not None:
#         # Create a bar chart
#         labels = ['Logistic Regression', 'Random Forest']
#         accuracies = [lr_accuracy, rf_accuracy]

#         plt.bar(labels, accuracies)
#         plt.ylabel('Accuracy')
#         plt.title('Comparison of Algorithms')
#         plt.ylim([0, 1])

#         # Convert the chart to a base64-encoded image for embedding in HTML
#         buffer = BytesIO()
#         plt.savefig(buffer, format='png')
#         buffer.seek(0)
#         chart_data = base64.b64encode(buffer.read()).decode()
#         buffer.close()

#         # Pass the results and chart data to the HTML template
#         return render_template('comparison.html', lr_accuracy=lr_accuracy, rf_accuracy=rf_accuracy, chart_data=chart_data)
#     else:
#         return "Please run both algorithms first."


# Final predictions 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from flask import Flask, render_template 

# Load and preprocess your dataset
file_paths = [
    "Dataset/1358073.csv", "Dataset/1358075.csv", "Dataset/1358076.csv", "Dataset/1358078.csv",
    "Dataset/1358083.csv", "Dataset/1358084.csv", "Dataset/1358087.csv", "Dataset/1358089.csv",
    "Dataset/1358092.csv", "Dataset/1367985.csv", "Dataset/1368002.csv", "Dataset/1377746.csv",
    "Dataset/1377751.csv", "Dataset/1377754.csv", "Dataset/1377759.csv", "Dataset/1377771.csv",
    "Dataset/1377774.csv", "Dataset/1388392.csv", "Dataset/1388398.csv"
]  # Update with your dataset file paths
dfs = [pd.read_csv(file_path) for file_path in file_paths]
merged_df = pd.concat(dfs)

# Data Preprocessing
merged_df = merged_df.drop(columns=['match_id', 'player_dismissed', 'other_wicket_type', 'other_player_dismissed', 'season', 'start_date'])  # Drop 'season' and 'start_date'

label_encoder = LabelEncoder()
merged_df['venue'] = label_encoder.fit_transform(merged_df['venue'])
merged_df['batting_team'] = label_encoder.fit_transform(merged_df['batting_team'])
merged_df['bowling_team'] = label_encoder.fit_transform(merged_df['bowling_team'])
merged_df['striker'] = label_encoder.fit_transform(merged_df['striker'])
merged_df['non_striker'] = label_encoder.fit_transform(merged_df['non_striker'])
merged_df['bowler'] = label_encoder.fit_transform(merged_df['bowler'])
merged_df['wicket_type'] = label_encoder.fit_transform(merged_df['wicket_type'])

X = merged_df.drop(columns=['runs_off_bat'])
y = merged_df['runs_off_bat']

@app.route('/predictions')
def predictions():

    # Initialize the imputer
    imputer = SimpleImputer(strategy='mean')

    # Fit and transform the imputer on your features
    X_imputed = imputer.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # Train and evaluate models
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)

    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    mse_dt = mean_squared_error(y_test, y_pred_dt)

    svm_model = SVR()
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    mse_svm = mean_squared_error(y_test, y_pred_svm)

    xgboost_model = XGBRegressor(random_state=42)
    xgboost_model.fit(X_train, y_train)
    y_pred_xgboost = xgboost_model.predict(X_test)
    mse_xgboost = mean_squared_error(y_test, y_pred_xgboost)

    # Calculate the prediction using the Random Forest model
    prediction = y_pred_rf  # Use Random Forest predictions here

    # Calculate the prediction using the Random Forest model
    prediction = y_pred_rf.tolist()  # Convert NumPy array to list

    # Create an accuracy graph
    algorithm_names = ['Random Forest', 'Decision Tree', 'Support Vector Machine', 'XGBoost']
    mse_scores = [mse_rf, mse_dt, mse_svm, mse_xgboost]


    plt.figure(figsize=(10, 6))
    plt.barh(algorithm_names, mse_scores, color='skyblue')
    plt.xlabel('Mean Squared Error')
    plt.title('Algorithm Comparison')   
    plt.gca().invert_yaxis()

    # Save the graph as a PNG image and convert it to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    graph_image = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    # Render the HTML template with the graph
    return render_template('predictions.html', mse_scores=mse_scores, algorithm_names=algorithm_names, prediction=prediction, graph_image=graph_image)


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)