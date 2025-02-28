
Dressify

**Your virtual clothing assistant!**
Dressify is a Streamlit-based application that categorizes clothing items, segments them from backgrounds, and allows you to create composite images with custom backgrounds.
You can see a demo for Dressify on my [Portfolio](https://hieungt02.my.canva.site)!

![result_image (3)](https://github.com/user-attachments/assets/49ffdab5-6733-4b88-a126-95c9d284da93)


---
Prerequisites
- This program was written with Python 3.11.
- Run the following command to install the required libraries:

   pip install streamlit tensorflow pillow numpy opencv-python

The installation process might take up to a few minutes.

- Ensure the trained model file (`clothing_model2.keras`) is in the same directory as the script.
- Run this command to launch the application with Streamlit:

   streamlit run dressify.py

Streamlit should automatically run it on your browser.
In Terminal, Streamlit will output something similar to this:
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://172.17.0.162:8501

If the browser didn't open up for you, copy any link and open it on your browser. I usually use the Local URL.
Important note: If you are running the program on a Macbook, Safari sometimes can not display the program. In that case, copy the link and use Chrome or other browsers instead.
---
Folders will appear when you run the program. I have also included a test_pictures folder which include some pictures of clothing items for you to try out the project, and a few of the output
pictures that was generated using the program.

## Features

- Automatically classify clothing as "Tops" or "Bottoms."
- Segment clothing from the background using color thresholds.
- Save segmented clothing images to categorized folders.
- Upload background images and place segmented clothing on them.
- Customize position, scale, and rotation of clothing on the background.
- Download result images.

## Step-by-Step Instructions on how to use the program

1. **Upload a Clothing Image**:
   - Click **Browse files** to upload an image (JPG, JPEG, or PNG).
   - The app will display the uploaded image, predict its category, and segment the clothing item.
2. **Save the Segmented Image**:
   - Click **Save to Collection** to save the segmented clothing item in its respective category folder (`categorized_uploads/Tops` or `categorized_uploads/Bottoms`).
3. **Upload a Background Image**:
   - Click **Browse files** to upload a background image.
   - Use sliders to adjust the position, scale, and rotation of the clothing.
4. **Download the Composite Image**:
   - Click **Download image** to save the final composite image.


## Files
- `uploaded_images`: Stores uploaded clothing images. (This folder will appear when you run the program)
- `categorized_uploads`: Contains categorized segmented images. (This folder will appear when you run the program)
- `clothing_model2.keras`: The pre-trained TensorFlow model for clothing classification.


## Notes

- The TensorFlow model (`clothing_model2.keras`) should be trained to classify clothing items (e.g., Tops vs. Bottoms).
- Images are normalized and resized to fit the model's input requirements.
- This application uses OpenCV for image segmentation and PIL for handling images.


Enjoy using Dressify!

##Information about the CNN model:
CNN Model Information:

Architecture:
Input: Image size (60x80x3).
Three convolutional layers with ReLU activation and filters increasing from 32 to 128, each followed by MaxPooling2D.
Fully connected dense layer with 128 neurons and a Dropout layer for regularization.
Output layer with 1 neuron and sigmoid activation for binary classification.

Training Details:
Optimizer: Adam.
Loss Function: Binary Crossentropy.
Epochs: 15.
Dataset: 262 training, 37 validation, 26 test samples.

Performance:
Validation Accuracy: ~89%.
Test Accuracy: ~92%.
