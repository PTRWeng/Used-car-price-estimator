import pandas as pd

# Load dataset
df = pd.read_csv("used_cars_UK.csv")

# Rename columns for easier access
df.rename(columns={
    "Price": "price",
    "Mileage(miles)": "mileage",
    "Registration_Year": "year",
    "Previous Owners": "previous_owners",
    "Fuel type": "fuel_type",
    "Body type": "body_type",
    "Engine": "engine",
    "Gearbox": "gearbox",
    "Doors": "doors",
    "Seats": "seats",
    "Emission Class": "emission_class",
    "Service history": "service_history"
}, inplace=True)

# Drop unnecessary columns
df.drop(columns=["title", "engine", "emission_class", "service_history"], inplace=True)

# Fill missing values
df["mileage"].fillna(df["mileage"].mean(), inplace=True)
df["previous_owners"].fillna(df["previous_owners"].mode()[0], inplace=True)
df["doors"].fillna(df["doors"].mode()[0], inplace=True)
df["seats"].fillna(df["seats"].mode()[0], inplace=True)
df.dropna(subset=["price", "year", "fuel_type"], inplace=True)

# Convert categorical data to numeric
df = pd.get_dummies(df, columns=["fuel_type", "body_type", "gearbox"], drop_first=True)

# Save cleaned dataset
df.to_csv("cleaned_used_cars.csv", index=False)
print("Data Preprocessing Completed! Cleaned dataset saved.")
