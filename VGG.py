import numpy as np
import cv2
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load the digits dataset from sklearn
digits = datasets.load_digits()
X = digits.images
y = digits.target

# Resize images to 64x64 and convert them to 3 channels (as VGG16 expects 3-channel inputs)
X_resized = np.array([cv2.resize(img, (64, 64)) for img in X])
X_resized = np.stack([X_resized] * 3, axis=-1)

# Normalize the images to be between 0 and 1
X_resized = X_resized.astype('float32') / 16.0

# One-hot encode the labels
encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resized, y_encoded, test_size=0.2, random_state=42)

# Load the pre-trained VGG16 model without the top fully connected layers
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Freeze the convolutional base
for layer in vgg16.layers:
    layer.trainable = False

# Add custom layers on top of VGG16
x = Flatten()(vgg16.output)
x = Dense(256, activation='relu')(x)
x = Dense(10, activation='softmax')(x)

# Create the final model
model = Model(inputs=vgg16.input, outputs=x)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")