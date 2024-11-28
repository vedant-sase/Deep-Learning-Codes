import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load dataset
# Use your dataset here, for example a CSV file with BTC-USD data
data = pd.read_csv("D:/bitcoin dataset/BTC-USD.csv")
  # Ensure columns have Date, Open, High, Low, Close, Volume, etc.

# Preprocess data
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Use Close price for prediction
prices = data[['Close']].values

# Scale data to range (0, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prices)

# Define training data
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Create datasets
time_step = 60  # Using 60 days of historical data to predict the next day
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split into training and testing data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build RNN model
model = Sequential()
model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(SimpleRNN(units=50))
model.add(Dense(units=1))  # Output layer for price prediction

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions to original scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))
plt.plot(data.index[:len(train_predict)], y_train, color="blue", label="Train Data")
plt.plot(data.index[len(train_predict):len(train_predict) + len(test_predict)], y_test, color="red", label="Actual Price")
plt.plot(data.index[len(train_predict):len(train_predict) + len(test_predict)], test_predict, color="green", label="Predicted Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
