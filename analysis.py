import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import joblib

# Load the dataset
data = pd.read_csv("player_performance.csv")

# ... Other data preprocessing ...

# Build decision tree classifiers
clf_batting = DecisionTreeClassifier()
clf_bowling = DecisionTreeClassifier()

# ... Train the classifiers ...


# Function to analyze player performance
def analyze_player_performance(player, pitch_type, weather):
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


# ... Other functions ...
