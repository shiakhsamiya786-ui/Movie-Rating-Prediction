import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\Shaikh Samiya\Downloads\IMDb Movies India.csv")

# Display first 5 rows
print(df.head())

# Dataset information
print(df.info())

# Missing values
print(df.isnull().sum())

# Remove missing values
df = df.dropna()

# Remove commas from Votes column and convert to numeric
df["Votes"] = df["Votes"].astype(str).str.replace(",", "")
df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")

# Remove ' min' from Duration and convert to numeric
df["Duration"] = df["Duration"].astype(str).str.replace(" min", "", regex=False)
df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")

# Drop missing values again
df = df.dropna()

# Encode categorical columns
le = LabelEncoder()

for col in ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3", "Name"]:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df[["Year", "Duration", "Votes", "Genre", "Director", "Actor 1", "Actor 2", "Actor 3", "Name"]]
y = df["Rating"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\nMean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Comparison
result = pd.DataFrame({
    "Actual Rating": y_test.values,
    "Predicted Rating": y_pred
})

print("\nFirst 10 Predictions:")
print(result.head(10))

# Plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Movie Rating Prediction")
plt.show()
