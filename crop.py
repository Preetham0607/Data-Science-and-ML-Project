import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv('crop_production_data.csv')  # Reading the CSV file

# Selecting features and target
X = data[['Rainfall (mm)', 'Temperature (°C)', 'Soil Quality (pH)']]
y = data['Crop Yield (kg/ha)']

# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a Random Forest model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predicting and evaluating
y_pred = model.predict(X_test)
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
