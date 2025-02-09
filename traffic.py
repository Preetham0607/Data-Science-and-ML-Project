import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
data = pd.read_csv('traffic_data.csv')

# Preparing features and target
X = np.array(data[['Hour', 'Weekend']])
y = np.array(data['Traffic Density (Vehicles/hour)'])

# Reshape data for LSTM
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Building LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(1, 2)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Training the model
model.fit(X, y, epochs=50, verbose=1)

# Save the model
model.save('traffic_forecasting_model.h5')
