# Import the necessary libraries.
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm # tqdm is used to display the progress bar while extracting features.
import pickle

# Load the ResNet50 model pre-trained on ImageNet.
model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))

# Set the model to be non-trainable.
model.trainable = False

# Create a sequential model that consists of the ResNet50 model followed by a GlobalMaxPooling2D layer.
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Define a function to extract features from an image.
def extract_features(img_path,model):
    """
    Extracts features from an image using the given model.

    Args:
        img_path (str): The path to the image.
        model (tensorflow.keras.Model): The model to use for feature extraction.

    Returns:
        np.array: The extracted features.
    """

    # Load the image.
    img = image.load_img(img_path,target_size=(224,224))

    # Convert the image to an array.
    img_array = image.img_to_array(img)

    # Expand the dimensions of the image array.
    expanded_img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image array.
    preprocessed_img = preprocess_input(expanded_img_array)

    # Predict the features of the image.
    result = model.predict(preprocessed_img).flatten()

    # Normalize the features.
    normalized_result = result / norm(result)

    return normalized_result

# Get the list of filenames of the images in the `images` directory.
filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

# Create a list to store the extracted features.
feature_list = []

# For each image in the `filenames` list, extract its features and add them to the `feature_list` list.
for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))

# Save the extracted features to a file called `embeddings.pkl`.
pickle.dump(feature_list,open('embeddings.pkl','wb'))

# Save the filenames to a file called `filenames.pkl`.
pickle.dump(filenames,open('filenames.pkl','wb'))