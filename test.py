import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam, Adagrad, RMSprop, Nadam
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import time

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# List of optimizers to test
optimizers = {
    'SGD': SGD(),
    'Adam': Adam(),
    'Adagrad': Adagrad(),
    'RMSprop': RMSprop(),
    'Nadam': Nadam()
}

# Create a dictionary to store results
results = {}

# Loop through the optimizers and train the model
for optimizer_name, optimizer in optimizers.items():
    print(f"Training with {optimizer_name} optimizer...")

    # Create and compile the model with the current optimizer
    model = create_model()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model and measure the time
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test), verbose=2)
    end_time = time.time()

    # Record the results
    elapsed_time = end_time - start_time
    results[optimizer_name] = {
        'history': history,
        'time': elapsed_time
    }

# Plot accuracy and loss for each optimizer
plt.figure(figsize=(12, 6))

# Plot Accuracy
plt.subplot(1, 2, 1)
for optimizer_name, result in results.items():
    plt.plot(result['history'].history['val_accuracy'], label=f'{optimizer_name} Val Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
for optimizer_name, result in results.items():
    plt.plot(result['history'].history['val_loss'], label=f'{optimizer_name} Val Loss')
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Print training times for each optimizer
for optimizer_name, result in results.items():
    print(f"Training with {optimizer_name} took {result['time']:.2f} seconds")
