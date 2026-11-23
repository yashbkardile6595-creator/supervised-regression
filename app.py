"""Predict the price of a sample house using the trained model."""
import pickle

import pandas as pd

# Load model (run bangalore_model.py first to create model.pkl)
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Example input
sample_data = pd.DataFrame(
    {
        "location": ["Whitefield"],
        "total_sqft": [1200],
        "bath": [2],
        "balcony": [1],
        "BHK": [2],
    }
)

# Predict price
prediction = model.predict(sample_data)
print("Predicted House Price:", prediction[0], "Lakhs")
