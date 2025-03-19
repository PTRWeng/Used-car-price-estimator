import pandas as pd

# Load the dataset (modify the filename if needed)
df = pd.read_csv("used_cars.csv")

# Display first few rows
print(df.head())

# Remove duplicates and missing values
df = df.drop_duplicates().dropna()

# Convert categorical variables (e.g., fuel type, brand) into numbers
df = pd.get_dummies(df, columns=['brand', 'fuel_type'])

# Save cleaned data
df.to_csv("cleaned_used_cars.csv", index=False)
print("Data Preprocessing Completed!")