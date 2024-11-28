import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Fix for UnicodeEncodeError: Set UTF-8 encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')

# Suppress TensorFlow AVX2/FMA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load and preprocess the MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten the images for the autoencoder
x_train = x_train.reshape((x_train.shape[0], -1))  # (60000, 784)
x_test = x_test.reshape((x_test.shape[0], -1))    # (10000, 784)

# Define dimensions
input_dim = x_train.shape[1]  # 28*28 = 784 for MNIST
encoding_dim = 64  # Dimension of the encoded representation

# Build the encoder
input_img = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_img)

# Build the decoder
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Combine into an autoencoder model
autoencoder = Model(input_img, decoded)

# Create encoder model
encoder = Model(input_img, encoded)

# Create decoder model
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder without logs (verbose=0) to avoid UnicodeEncodeError
autoencoder.fit(
    x_train, x_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test),
    verbose=0  # Suppresses progress bar and avoids encoding errors
)

# Make predictions (also suppress logs to avoid Unicode issues)
encoded_imgs = encoder.predict(x_test, verbose=0)
decoded_imgs = decoder.predict(encoded_imgs, verbose=0)

# Visualize original and reconstructed images
n = 10  # Number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')
    
    # Display reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

plt.tight_layout()
plt.show()
