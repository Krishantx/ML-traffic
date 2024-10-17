import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import argparse

def run(args):
    model_path = args.model_path
    
    print("----------- Starting Streamlit App -----------")
    st.title("Traffic Sign Recognition App")
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a traffic sign image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).resize((30, 30))
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess the image
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Predict the class
        prediction = model.predict(image_array)
        
        
        predicted_class = np.argmax(prediction, axis=1)[0]
        class_name = "Speed Limit"
        if predicted_class <= 5 and predicted_class == 7 and predicted_class == 8:
            class_name = "Speed Limit"
        elif predicted_class == 6:
            class_name = "End of Speed limit"
        elif predicted_class == 9:
            class_name = "No Passing"
        elif predicted_class == 10:
            class_name = "No Passing for vehicles over 3.5 tonnes"
        elif predicted_class == 11:
            class_name = "Priority"
        elif predicted_class == 12:
            class_name = "Priority Road"
        elif predicted_class == 13:
            class_name = "Yield"
        elif predicted_class == 14:
            class_name = "Stop"
        elif predicted_class == 15:
            class_name = "Road Closed"
        elif predicted_class == 16:
            class_name = "Vehicle over 3.5 tonnes prohibited"
        elif predicted_class == 17:
            class_name = "Do not enter"
        elif predicted_class == 18:
            class_name = "General danger"
        elif predicted_class == 19:
            class_name = "Left curve"
        elif predicted_class == 20:
            class_name = "Right Curve"
        elif predicted_class == 21:
            class_name = "Double Curve"
        elif predicted_class == 22:
            class_name = "Uneven road surface"
        elif predicted_class == 23:
            class_name = "Slippery when wet"
        elif predicted_class == 24:
            class_name = "Road Narrows"
        elif predicted_class == 25:
            class_name = "Men at work"
        elif predicted_class == 26:
            class_name = "Traffic signal Ahead"
        elif predicted_class == 27:
            class_name = "Pedisteran Crossing"
        elif predicted_class == 28:
            class_name = "Watch for children"
        elif predicted_class == 29:
            class_name = "Bicycle crossing"
        elif predicted_class == 30:
            class_name = "Ice/Snow"
        elif predicted_class == 31:
            class_name = "Wild Animal Crossing"
        elif predicted_class == 32:
            class_name = "End of all restrictions"
        elif predicted_class == 33:
            class_name = "Turn Right Ahead"
        elif predicted_class == 34:
            class_name = "Turn Left Ahead"
        elif predicted_class == 35:
            class_name = "Ahead Only"
        elif predicted_class == 36:
            class_name = "Ahead or turn right only"
        elif predicted_class == 37:
            class_name = "Ahead or turn left only"
        elif predicted_class == 38:
            class_name = "Pass by On Rigt"
        elif predicted_class == 39:
            class_name = "Pass by On Left"
        elif predicted_class == 40:
            class_name = "Roundabout"
        elif predicted_class == 41:
            class_name = "End of No Passing"
        elif predicted_class == 42:
            class_name = "End of No Passing for trucks"
        st.write(f"Predicted Class: {class_name}")
        print(f"Predicted Class: {predicted_class}")
        print(predicted_class)
    
    print("----------- Streamlit App Running -----------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streamlit App for Traffic Sign Recognition")
    parser.add_argument('--model_path', type=str, default='models/best_model.h5', help='Path of the trained model')
    args = parser.parse_args()
    run(args)
