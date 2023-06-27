import csv


def get_player_stats(player_name):
    with open("player_performance.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["Player"] == player_name:
                # Construct the image filename based on the player's name
                image_filename = row["Player"].replace(" ", "_") + ".jpg"

                return {
                    "Player": row["Player"],
                    "Matches": row["Matches"],
                    "Innings": row["Innings"],
                    "Runs": row["Runs"],
                    "Wickets": row["Wickets"],
                    "Batting_Average": row["Batting_Average"],
                    "Bowling_Average": row["Bowling_Average"],
                    "Strike_Rate": row["Strike_Rate"],
                    "Economy_Rate": row["Economy_Rate"],
                    "Image": image_filename,  # Add the image filename to the player's stats
                }
    return None
