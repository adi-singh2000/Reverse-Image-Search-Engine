import pickle
import time
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2

# Load the feature vectors from the embeddings.pkl file
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))

# Load the corresponding filenames from the filenames.pkl file
filenames = pickle.load(open('filenames.pkl', 'rb'))
print(feature_list[0])

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Create a Sequential model with ResNet50 as the base and add GlobalMaxPooling2D layer
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Load and preprocess the query image
img = image.load_img('static/uploads/1163.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)

# Extract features from the query image using the ResNet50 model
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)
print(normalized_result)

start = time.time()

# Initialize a NearestNeighbors object with the feature_list and fit it
neighbors = NearestNeighbors(n_neighbors=5,  metric='euclidean')
neighbors.fit(feature_list)

# Find the nearest neighbors for the query image's feature vector
distances, indices = neighbors.kneighbors([normalized_result])

end = time.time()
print("KNN took {:.2f} seconds".format(end - start))

# Get the labels of the top 5 images
top_5_labels = [filenames[i] for i in indices[0]]

# Compare the labels of the top 5 images to the label of the query image
correct_label = filenames[indices[0][0]]

# Calculate the accuracy of the reverse image search
accuracy = (top_5_labels.count(correct_label) / len(top_5_labels)) * 100

print("Accuracy: {:.2f}%".format(accuracy))

# Display the retrieved similar images
for file in indices[0][1:6]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output', cv2.resize(temp_img, (512, 512)))
    cv2.waitKey(0)

########## ANNOY - If your dataset contain millions of images then use annoy rather than KNN. ##########      
        
# from annoy import AnnoyIndex  
# ## create Annoy Index
# num_trees = 100
# num_features = result.shape[0]

# start = time.time()
# index = AnnoyIndex(num_features, metric='manhattan')
# for i in range(len(feature_list)):
#     index.add_item(i, feature_list[i])
# index.build(num_trees)

# n_neighbors = 5
# indices = index.get_nns_by_vector(normalized_result, n_neighbors, include_distances=False)
# end = time.time()
# print("Annoy took {:.2f} seconds".format(end - start))

# for file in indices:
#         temp_img = cv2.imread(filenames[file])
#         cv2.imshow('output',cv2.resize(temp_img,(512,512)))
#         cv2.waitKey(0)