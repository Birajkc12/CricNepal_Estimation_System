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

# Initialize the imputer
imputer = SimpleImputer(strategy='mean')

# Fit and transform the imputer on your features
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Create an accuracy graph
algorithm_names = ['Random Forest', 'Decision Tree', 'Support Vector Machine', 'XGBoost']
mse_scores = [mse_rf, mse_dt, mse_svm, mse_xgboost]

plt.figure(figsize=(10, 6))
plt.barh(algorithm_names, mse_scores, color='skyblue')
plt.xlabel('Mean Squared Error')
plt.title('Algorithm Comparison')
plt.gca().invert_yaxis()
plt.show()
