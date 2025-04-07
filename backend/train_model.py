import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load cleaned dataset
df = pd.read_csv("cleaned_used_cars.csv")

# Define input features and target
X = df.drop(columns=["price"])  # Features
y = df["price"]                 # Target

# Split into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Model Evaluation:\nMSE: {mse:.2f}, R²: {r2:.2f}")

# Save the trained model to a file
joblib.dump(model, "car_price_model.pkl")
print("Model trained and saved as 'car_price_model.pkl'")
