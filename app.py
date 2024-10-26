import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Add custom CSS for dark blue background
st.markdown(
    """
    <style>
    .stApp {
        background-color: #00008B;  /* Dark blue color */
    }
    .stTextInput>div>input {
        color: #FFFFFF;  /* Input text color (white for better contrast) */
        font-size: 20px;  /* Input font size */
    }
    .stButton>button {
        background-color: #4CAF50;  /* Button color */
        color: white;  /* Button text color */
        font-size: 18px;  /* Button font size */
    }
    </style>
    """, unsafe_allow_html=True
)

# Load the trained model
model = load_model('brain_tumor_model.h5')

def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to match model's expected input
    img_array = np.array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

st.title("Brain Tumor Classification")
st.subheader("Upload your MRI image below:")

uploaded_file = st.file_uploader("Upload an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    img_array = preprocess_image(image)
    
    if st.button("Classifier Tumor"):
        prediction = model.predict(img_array)
        if prediction[0][0] > 0.5:
            st.write("Prediction: Tumor detected.")
        else:
            st.write("Prediction: No tumor detected.")
