import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('dataNepal.csv')

# Convert categorical data to numerical data using one-hot encoding
data_encoded = pd.get_dummies(
    data, columns=['Opponent', 'Location', 'Toss Winner'])

# Split the dataset into training and testing sets
X = data_encoded.drop(['Result', 'Score'], axis=1)
y = data_encoded['Result']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train the logistic regression model with increased max_iter
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Make predictions on the testing set and evaluate the model's performance
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Make predictions on new data
new_data = pd.DataFrame({
    'Opponent': ['Bangladesh'],
    'Location': ['Home'],
    'Toss Winner': ['Bangladesh'],
    'Opponent_Strength': [7],
    'Location_Factor': [1.3],
    'Toss_Winner_Factor': [1.0]
})
new_data_encoded = pd.get_dummies(
    new_data, columns=['Opponent', 'Location', 'Toss Winner'])
new_X = new_data_encoded.reindex(columns=X_train.columns, fill_value=0)
prediction = clf.predict(new_X)
print('Prediction:', prediction)

# Estimate the range of total runs scored by Team Nepal in the next match
total_runs_mean = int(data['Total Runs'].mean())
total_runs_std = int(data['Total Runs'].std())

# Calculate the estimated range using the added parameters
range_multiplier = 0.2  # Adjust this value to decrease the range further
range_low = int(total_runs_mean - total_runs_std * range_multiplier)
range_high = int(total_runs_mean + total_runs_std * range_multiplier)

# Consider the impact of added parameters on the range estimation
range_low += int(new_data['Opponent_Strength'].values[0] * new_data['Location_Factor'].values[0])
range_high += int(new_data['Opponent_Strength'].values[0] * new_data['Location_Factor'].values[0])

print('Estimated range of total runs:', range_low, '-', range_high)
