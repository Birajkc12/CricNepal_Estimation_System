from flask import Flask, render_template, request, session, redirect
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Import DecisionTreeRegressor from sklearn.tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from playerprofile import get_player_stats
from werkzeug.utils import secure_filename
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
import pickle
import csv
import joblib
from predict import make_prediction


app = Flask(__name__)
app.secret_key = "64131316905e8dea79bc77eda222e351"

# Load the dataset
data = pd.read_csv("player_performance.csv")

data["Image"] = data["Player"].str.replace(" ", "_") + ".jpg"

# Calculate additional metrics
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


import pandas as pd

# Load the initial dataset from the CSV file
data = pd.read_csv("player_performance.csv")

# ... other code ...


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


# # Route for prediction
# @app.route("/predict", methods=["POST"])
# def predict():
#     # Retrieve the form data
#     matches = int(request.form["matches"])
#     innings = int(request.form["innings"])
#     runs = int(request.form["runs"])
#     wickets = int(request.form["wickets"])
#     balls_faced = int(request.form["balls_faced"])

#     # Prepare the data for training
#     X = data[
#         [
#             "Matches",
#             "Innings",
#             "Runs",
#             "Wickets",
#             "Batting_Average",
#             "Bowling_Average",
#             "Strike_Rate",
#             "Economy_Rate",
#         ]
#     ]

#     try:
#         y = data[
#             "Above_Avg_Batting"
#         ]  # Replace "Above_Avg_Batting" with the actual target variable column name
#     except KeyError as e:
#         return f"Error: {e}. Please check the column name for the target variable in your dataset."

#     # Train the linear regression model
#     model = LinearRegression()
#     model.fit(X, y)

#     # Save the trained model
#     joblib.dump(model, "trained_model.pkl")

#     # Perform the prediction
#     model = joblib.load("trained_model.pkl")  # Load the trained model

#     X_input = [
#         [matches, innings, runs, wickets, balls_faced]
#     ]  # Format the input features

#     try:
#         y_pred = model.predict(X_input)  # Make the prediction
#     except Exception as e:
#         return f"Error occurred during prediction: {e}"

#     # Format and return the prediction result
#     prediction_result = f"The predicted value is: {y_pred[0]}"
#     return render_template("prediction.html", prediction=prediction_result)


# Route for the prediction
@app.route("/predict", methods=["POST"])
def predict():
    opponent = request.form["opponent"]
    location = request.form["location"]
    toss_winner = request.form["toss_winner"]

    prediction = make_prediction(opponent, location, toss_winner)

    return render_template("prediction.html", prediction=prediction)


# Load player data from CSV file
def load_player_data(filename):
    players = []
    with open(filename, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            players.append(row)
    return players


# Load ICC ranking data from a similar CSV file if needed
# icc_ranking = ...


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


# ...

# ...


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


def analyze_player_performance(player, pitch_type, weather, algorithm):
    if algorithm == "linear":
        prediction = predict_linear_regression(player, pitch_type, weather)
    else:
        prediction = predict_decision_tree(player, pitch_type, weather)
        return prediction  # Return the prediction dictionary directly

    return render_template(
        "analysis_result.html",
        prediction=prediction,
        player_image=player["Image"],
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

if __name__ == "__main__":
    app.run(debug=True)
