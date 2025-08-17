import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image


# Step 1: Load the pre-trained MobileNetV2 model
def load_model():
    model = MobileNetV2(weights="imagenet")
    return model


# Step 2: Preprocess the image for MobileNetV2
def preprocess_image(image):
    img = np.array(image.convert("RGB"))  
    img = cv2.resize(img, (224, 224))  
    img = preprocess_input(img)  
    img = np.expand_dims(img, axis=0)  
    return img


# Step 3: Classify uploaded image
def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None


# Step 4: Build Streamlit App
def main():
    st.set_page_config(page_title="AI Image Classifier", page_icon="🖼️", layout="centered")

    st.title("🖼️ AI Image Classifier")
    st.write("Upload an image and let AI tell you what is in it!")

    # Cache the model so it doesn’t reload every time
    @st.cache_resource
    def load_cached_model():
        return load_model()

    model = load_cached_model()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Show uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Button for classification
        if st.button("🔍 Classify Image"):
            try:
                pil_image = Image.open(uploaded_file)

                with st.spinner("Analyzing Image..."):
                    predictions = classify_image(model, pil_image)

                if predictions:
                    st.subheader("📌 Predictions")
                    for _, label, score in predictions:
                        st.progress(min(100, int(score * 100)))
                        st.write(f"**{label}** → {score:.2%}")

            except Exception as e:
                st.error(f"Error: {str(e)}")


# Step 5: Run app
if __name__ == "__main__":
    main()
