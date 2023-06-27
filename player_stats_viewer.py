from playerprofile import get_player_stats

# Example usage:
player_name = input("Enter player name: ")
player_stats = get_player_stats(player_name)
if player_stats:
    print(f"Player: {player_stats['Player']}")
    print(f"Matches: {player_stats['Matches']}")
    print(f"Innings: {player_stats['Innings']}")
    # Display the rest of the player's stats as needed
else:
    print("Player not found.")
