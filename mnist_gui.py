import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, datasets
from PIL import Image

# Function to load and preprocess MNIST data
def load_mnist_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images = np.expand_dims(train_images, -1) / 255.0
    test_images = np.expand_dims(test_images, -1) / 255.0
    return (train_images, train_labels), (test_images, test_labels)

# Function to load and preprocess CIFAR-10 data
def load_cifar_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return (train_images, train_labels), (test_images, test_labels)

# Function to create a model based on the selected dataset
def create_model(input_shape, dataset_name):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    
    if dataset_name == "MNIST":
        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dense(16, activation="relu"))
    elif dataset_name == "CIFAR-10":
        model.add(layers.Dense(512, activation="relu"))
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dense(32, activation="relu"))

    model.add(layers.Dense(10, activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Function to preprocess the uploaded image
def preprocess_image(uploaded_image, dataset_name):
    img = Image.open(uploaded_image)
    img = img.resize((28, 28)) if dataset_name == "MNIST" else img.resize((32, 32))

    # Convert image to grayscale for MNIST or RGB for CIFAR
    if dataset_name == "MNIST":
        img = img.convert("L")  # Grayscale for MNIST
        img = np.expand_dims(np.array(img), -1) / 255.0
    else:
        img = np.array(img) / 255.0

    return np.expand_dims(img, axis=0)  # Add batch dimension

# Streamlit App
st.title("Image Classification with MNIST and CIFAR-10")
st.write("Choose a dataset, train a model, and upload an image for classification.")

# Dataset selection
dataset_name = st.selectbox("Select Dataset", ["MNIST", "CIFAR-10"])

# Load data based on selection
if dataset_name == "MNIST":
    input_shape = (28, 28, 1)
    (train_images, train_labels), (test_images, test_labels) = load_mnist_data()
elif dataset_name == "CIFAR-10":
    input_shape = (32, 32, 3)
    (train_images, train_labels), (test_images, test_labels) = load_cifar_data()

# Display shapes of data
st.write(f"Training data shape: {train_images.shape}")
st.write(f"Test data shape: {test_images.shape}")

# Train Model button
if st.button("Train Model"):
    model = create_model(input_shape, dataset_name)
    st.write("Training the model...")

    # Train the model
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

    # Store the trained model in session state
    st.session_state.model = model

    # Display training and validation accuracy
    st.write("Training complete!")
    st.write("Final Training Accuracy:", history.history['accuracy'][-1])
    st.write("Final Validation Accuracy:", history.history['val_accuracy'][-1])

    # Plot training and validation accuracy over epochs
    st.line_chart({
        "Training Accuracy": history.history['accuracy'],
        "Validation Accuracy": history.history['val_accuracy']
    })

# Test Model button
if st.button("Test Model"):
    if 'model' in st.session_state:
        # If model exists in session state, use it for evaluation
        test_loss, test_accuracy = st.session_state.model.evaluate(test_images, test_labels)
        st.write("Test Accuracy:", test_accuracy)
    else:
        st.write("Please train the model first.")

# Image Upload Section
st.subheader("Upload an Image for Classification")

uploaded_image = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Preprocess the image
    preprocessed_image = preprocess_image(uploaded_image, dataset_name)

    # Predict using the trained model
    if 'model' in st.session_state:
        prediction = st.session_state.model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction)
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.write(f"Prediction: Class {predicted_class}")
    else:
        st.write("Please train the model first.")
