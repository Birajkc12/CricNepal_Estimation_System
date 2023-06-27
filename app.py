from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import os
from playerprofile import get_player_stats
from flask import Flask, session, redirect


app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load the dataset
data = pd.read_csv("player_performance.csv")

data["Image"] = data["Player"].str.replace(" ", "_") + ".jpg"


# Calculate additional metrics
data["Above_Avg_Batting"] = data["Batting_Average"] > data["Batting_Average"].mean()
data["Above_Avg_Bowling"] = data["Bowling_Average"] > data["Bowling_Average"].mean()

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
        batting_predictions | filtered_data["Above_Avg_Batting"]
    ]
    above_average_bowling = filtered_data.loc[
        bowling_predictions | filtered_data["Above_Avg_Bowling"]
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


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # Perform authentication logic here (e.g., validate username and password)
        # You can use a database or store admin credentials in a secure manner

        # If authentication is successful, store the admin's session
        session["admin"] = True
        return redirect("/admin")

    return render_template("admin_login.html")


@app.route("/admin")
def admin_panel():
    # Check if the admin is logged in
    if not session.get("admin"):
        return redirect("/admin/login")

    # Render the admin panel HTML template
    return render_template("admin_panel.html")


# ...

if __name__ == "__main__":
    app.run(debug=True)
