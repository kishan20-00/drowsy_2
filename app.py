import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the pre-trained Keras model
best_model = load_model('detection.h5')  # Replace 'your_model_path.h5' with the path to your .h5 file

# Function to preprocess the input image
def preprocess_image(image):
    # Resize the image to match the size used during training (64x64)
    resized_img = image.resize((64, 64))
    # Convert image to grayscale
    grayscale_img = resized_img.convert('L')
    # Convert image to numpy array
    np_img = np.array(grayscale_img)
    # Normalize pixel values
    normalized_img = np_img / 255.0
    return normalized_img

# Function to predict using the loaded model
def predict_image(image):
    preprocessed_img = preprocess_image(image)
    # Add batch dimension
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
    # Reshape to match model input shape (4000, 64, 64, 1)
    preprocessed_img = preprocessed_img.reshape(1, 64, 64, 1)
    result = best_model.predict(preprocessed_img)
    return result

# Load and preprocess the image
image_path = 'open.jpg'  # Replace 'path_to_your_image.jpg' with the path to your image file
image = Image.open(image_path)
prediction = predict_image(image)

# Display the prediction result
if prediction > 0.5:
    print('Open')
else:
    print('Closed')
