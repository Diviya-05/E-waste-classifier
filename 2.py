import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your trained model
model = tf.keras.models.load_model("e_waste_classifier.h5")

# Path to a test image
img_path = "C:\\Users\\hp\\Downloads\\keyboard.jpeg"  

# Preprocess the image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize

# Predict
predictions = model.predict(img_array)
class_index = np.argmax(predictions[0])

# Get class labels from training data
class_labels = ['hazardous b', 'recyclable oth ', 'repairable mob', 'reusable key']  # or whatever your actual folder/class names are


print(f"🔍 Predicted category: {class_labels[class_index]}")
