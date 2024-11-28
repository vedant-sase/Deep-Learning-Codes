import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Correct model path - make sure this path is correct
MODEL_PATH = 'C:/Users/Vedant/Desktop/cats_and_dogs_classification.h5'  # Update if needed

# Load the pre-trained model with error handling
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()  # Stop the execution if the model cannot be loaded

# Streamlit user interface
st.title('Cat vs Dog Image Classification')
st.write('Upload an image of a cat or a dog, and the model will predict the class.')

# File uploader for image input
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Load and display the uploaded image
    img = image.load_img(uploaded_image, target_size=(150, 150))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Convert the image to a numpy array and normalize it
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict using the trained model
    prediction = model.predict(img_array)

    # Show the prediction result
    if prediction[0] > 0.5:
        st.write("Prediction: Dog")
    else:
        st.write("Prediction: Cat")

