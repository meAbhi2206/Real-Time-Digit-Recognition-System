import cv2
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# Load the trained model
model = tf.keras.models.load_model('mnist_model.h5')

# Function to preprocess the drawn image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 28, 28, 1))
    return reshaped

# Streamlit app
st.set_page_config(layout="wide")
st.title("Real-time Digit Recognition")
st.write("Draw a digit on the canvas and click 'Predict' to see the result.")

# Create a resizable canvas for drawing
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=st.sidebar.slider("Stroke Width", 5, 30, 20),
    stroke_color="white",
    background_color="black",
    height=400,
    width=400,
    drawing_mode="freedraw",
    key="canvas",
)

# Real-time prediction
col1, col2 = st.columns([1, 1])
if col2.button("Predict"):
    # Preprocess the drawn image
    image_data = canvas_result.image_data.astype(np.uint8)
    image = cv2.cvtColor(image_data, cv2.COLOR_RGBA2BGR)
    preprocessed_image = preprocess_image(image)

    # Make a prediction using the trained model
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)

    # Display the predicted class
    with col1:
        st.subheader("Prediction")
        digit_image = image[:, :, 0]
        st.image(digit_image, width=200)
        st.write(f"Predicted Class: {predicted_class}")

        # Visualize prediction probabilities
        st.subheader("Prediction Probabilities")
        probabilities = prediction[0]
        labels = [str(i) for i in range(10)]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, probabilities, color="purple", alpha=0.7)
        ax.set_xlabel("Digit")
        ax.set_ylabel("Probability")
        ax.set_ylim([0, 1])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)
        st.pyplot(fig)

    with col2:
        st.subheader("Canvas")
        st.image(canvas_result.image_data, width=200)
