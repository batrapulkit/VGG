import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Title
st.title("Image Classification App")

# Load model
@st.cache_resource
def load_vgg_model():
    model = load_model('VGG.h5')
    return model

model = load_vgg_model()

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Process image
    def preprocess_image(image_path):
        img = load_img(image_path, target_size=(224, 224))  # Adjust size based on your model
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalizing, adjust if necessary
        return img_array

    # Save temporarily and process
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
        processed_image = preprocess_image("temp_image.png")
    
    # Predict
    prediction = model.predict(processed_image)
    st.write("Prediction:", prediction)

    # Optional: Map predictions to class labels
    # Example:
    # labels = ["Class A", "Class B", "Class C"]
    # predicted_label = labels[np.argmax(prediction)]
    # st.write(f"Predicted Label: {predicted_label}")
else:
    st.write("Upload an image to get started.")
