"""Train a Random Forest model to predict Bangalore house prices (in Lakhs)."""
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Load dataset
df = pd.read_csv("datasets/bangalore_house_data.csv")

# Select useful columns and drop missing values
df = df[["location", "size", "total_sqft", "bath", "balcony", "price"]]
df = df.dropna()


def convert_sqft(x):
    """Convert total_sqft to float; ranges like '2100 - 2850' become NaN."""
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan


df["total_sqft"] = df["total_sqft"].apply(convert_sqft)
df = df.dropna()

# Extract BHK number from the 'size' column (e.g. "2 BHK" -> 2)
df["BHK"] = df["size"].apply(lambda x: int(x.split(" ")[0]))

# Features and target
X = df[["location", "total_sqft", "bath", "balcony", "BHK"]]
y = df["price"]

categorical_features = ["location"]
numerical_features = ["total_sqft", "bath", "balcony", "BHK"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", SimpleImputer(strategy="median"), numerical_features),
    ]
)

# Model pipeline
model = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# Save model (model.pkl is large, so it is git-ignored; re-create it by running this script)
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")
