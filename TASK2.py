import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the dataset
df = pd.read_csv('C:/Users/Sachin/Downloads/Clean_Dataset.csv/Flight_Dataset.csv', encoding='latin-1')

# Feature selection and engineering
X = df[['days_left', 'duration', 'departure_time', 'airline', 'source_city', 
        'destination_city', 'class']]
y = df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define feature types
numeric_features = ['days_left', 'duration']
categorical_features = ['departure_time', 'airline', 'source_city', 'destination_city', 'class']

# Create preprocessing steps
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Print metrics
print("\nModel Performance Metrics:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared Score: {r2:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")

# Visualizations
plt.figure(figsize=(10, 5))

# 1. Actual vs Predicted Values
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Flight Prices')

# 2. Residual Plot
residuals = y_test - y_pred
plt.subplot(1, 2, 2)
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.tight_layout()
plt.show()

# Print unique airlines in the dataset
print("\nUnique airlines in the dataset:")
print(df['airline'].unique())

# User input for flight details
print("\nPlease enter the flight details:")
airline = input("Airline (e.g., Air_India, SpiceJet): ").strip().title()
flight_class = input("Class (e.g., Economy, Business): ").strip().capitalize()

# Fixed values for other parameters
days_left = 30  # You can change this value as needed
duration = 2.0  # You can change this value as needed
departure_time = 'Morning'  # Set to a default value
source_city = 'Mumbai'  # Set to a default value
destination_city = 'Delhi'  # Set to a default value

# Example prediction with user input
example_data = pd.DataFrame({
    'days_left': [days_left],
    'duration': [duration],
    'departure_time': [departure_time],
    'airline': [airline],
    'source_city': [source_city],
    'destination_city': [destination_city],
    'class': [flight_class]
})

# Predict the price for the example data
predicted_price = model.predict(example_data)[0]
print(f"\nExample Prediction:")
print(f"For the given flight details:")
print(f"Predicted Price for '{airline}': ₹{predicted_price:.2f}")

# Feature importance analysis (for numeric features)
numeric_coef = model.named_steps['regressor'].coef_[:len(numeric_features)]
plt.figure(figsize=(8, 6))
feature_importance = pd.DataFrame({
    'Feature': numeric_features,
    'Coefficient': abs(numeric_coef)
})
sns.barplot(x='Feature', y='Coefficient', data=feature_importance)
plt.title('Numeric Feature Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
