import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import os
import numpy as np
import cv2
import tensorflow as tf
from io import BytesIO

# Center-align all buttons
st.markdown(
    """
    <style>
    .stButton > button {
        display: block;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Directories for saving images
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

#Folder to store categorized, segmented items
BASE_DIR = "categorized_uploads"
CATEGORIES = ["Bottoms", "Tops"]
for category in CATEGORIES:
    os.makedirs(os.path.join(BASE_DIR, category), exist_ok=True)

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("clothing_model2.keras")

model = load_model()

def preprocess_image(img_path, target_size=(60, 80)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

def segment_clothing(img: np.ndarray, margin: int = 5, threshold: int = 72):
    """
    Segments the clothing piece from the background based on the background color.

    Parameters:
        img (np.ndarray): Input image as a NumPy array (BGR format).
        margin (int): Number of pixels to consider around the edges for background color estimation.
        threshold (int): Threshold for color similarity.

    Returns:
        segmented_img (PIL.Image): The segmented clothing image with a transparent background.
    """
    top = img[:margin, :]
    bottom = img[-margin:, :]
    left = img[:, :margin]
    right = img[:, -margin:]

    # Calculate average background color
    margins = np.concatenate((top.flatten(), bottom.flatten(), left.flatten(), right.flatten()))
    margin_pixels = margins.reshape(-1, 3)
    background_color = np.mean(margin_pixels, axis=0)

    # Create binary mask
    binary_image_inv = cv2.inRange(img, background_color - threshold, background_color + threshold)
    binary_image = cv2.bitwise_not(binary_image_inv)

    # Apply mask to extract clothing
    mask = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    segmented = cv2.bitwise_and(img, mask)

    # Convert segmented image back to RGB
    segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)

    # Add alpha channel
    b, g, r = cv2.split(segmented_rgb)
    alpha = binary_image
    segmented_with_alpha = cv2.merge((b, g, r, alpha))

    return Image.fromarray(segmented_with_alpha)

# Streamlit App Title
st.title("Dressify")
st.markdown(
    """
    **Your virtual clothing assistant!**\n
    Upload your clothing item and having them automatically categorized. Style your outfits seamlessly.
    Create stunning visuals with customizable backgrounds.
    """
)
st.markdown("---")

#Upload clothing image
st.subheader("Upload clothing image")
uploaded_file = st.file_uploader("Upload your clothing image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    #Running picture of clothing to run through prediction model
    preprocessed_image = preprocess_image(img_path, target_size=(60, 80))
    predictions = model.predict(preprocessed_image)
    predicted_class = 1 if predictions[0] >= 0.5 else 0

    class_labels = {0: 'Bottoms', 1: 'Tops'}
    predicted_label = class_labels[predicted_class]

    display_image = Image.open(img_path)
    clothing_np = np.array(display_image.convert("RGBA"))
    clothing_bgr = cv2.cvtColor(clothing_np, cv2.COLOR_RGBA2BGR)
    segmented_clothing = segment_clothing(clothing_bgr)

    col1, col2 = st.columns(2)
    fixed_width = 300
    with col1:
        display_image.thumbnail((fixed_width, fixed_width))
        st.image(display_image, caption="Uploaded image")
        #st.write(f"Predicted Probability: {predictions[0][0]:.2f}")
        st.write(f"Predicted Category: **{predicted_label}**")

    with col2:
        segmented_clothing.thumbnail((fixed_width, fixed_width))
        st.image(segmented_clothing, caption="Segmented clothing")

    #Button to save segmented clothes to folder (categorized_uploads)
    if st.button("Save to Collection"):
        category_path = os.path.join(BASE_DIR, predicted_label)
        os.makedirs(category_path, exist_ok=True)
        segmented_file_path = os.path.join(category_path, f"segmented_{uploaded_file.name}")
        segmented_clothing.save(segmented_file_path, "PNG")
        st.success(f"Saved as: {segmented_file_path}")

    st.markdown("---")

    #Upload background image
    st.subheader("Choose background image")
    background_file = st.file_uploader("Upload your background image", type=["jpg", "jpeg", "png"])

    if background_file:
        background_image = Image.open(background_file).convert("RGBA")
        #st.image(background_image, caption="Uploaded Background Image", use_container_width=True)

        st.markdown("---")

        st.subheader("Adjust placement")
        col1, col2 = st.columns([1, 2])  # Adjust width ratios as needed

        with col1:
            x_pos = st.slider("X Position", 0, background_image.width, 50)
            y_pos = st.slider("Y Position", 0, background_image.height, 50)
            scale = st.slider("Size (Scale Factor)", 10, 300, 100)  # Percentage scale
            rotation = st.slider("Rotation (Degrees)", 0, 360, 0)

        with col2:
            #Resize
            clothing_resized = segmented_clothing.resize(
                (int(segmented_clothing.width * scale / 100), int(segmented_clothing.height * scale / 100))
            ).rotate(rotation, expand=True)

            #Ensure background has an alpha channel
            background_copy = background_image.copy()

            # Paste clothing onto the background
            background_with_clothing = background_copy.copy()
            background_with_clothing.paste(clothing_resized, (x_pos, y_pos), clothing_resized)

            #Display the final composite image
            st.image(background_with_clothing, caption="Result image", use_container_width=True)

            #Button to download the composite image
            buffer = BytesIO()
            background_with_clothing.save(buffer, format="PNG")
            buffer.seek(0)
            st.download_button(
                label="Download image",
                data=buffer,
                file_name="result_image.png",
                mime="image/png",
            )


