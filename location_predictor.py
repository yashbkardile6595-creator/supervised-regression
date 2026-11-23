"""Interactive house price prediction from user input."""
import pickle

import pandas as pd

# Load trained model (run bangalore_model.py first to create model.pkl)
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

print("===== Bangalore House Price Prediction =====")

location = input("Enter Bangalore Location: ")
total_sqft = float(input("Enter Total Square Feet: "))
bath = int(input("Enter Number of Bathrooms: "))
balcony = int(input("Enter Number of Balconies: "))
bhk = int(input("Enter BHK: "))

input_data = pd.DataFrame(
    {
        "location": [location],
        "total_sqft": [total_sqft],
        "bath": [bath],
        "balcony": [balcony],
        "BHK": [bhk],
    }
)

prediction = model.predict(input_data)
print("Estimated House Price:")
print(round(prediction[0], 2), "Lakhs")
