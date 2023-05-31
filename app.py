#Import necessary libraries
from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

#create flask application instance
app = Flask(__name__)

#load the features and filenames
feature_list = np.array(pickle.load(open( 'embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

#load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#function to extract features from an image
def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

#function to recommend similar products using KNN
def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

#configure the flask app settings for uploading files
UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 2048 * 2048
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'jfif', 'svg'])
 
#function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#route to home page
@app.route('/')
def home():
    return render_template('index.html')

#route to upload image and recommend similar images
@app.route('/', methods=['POST'])
def upload_to_recommend_product():
    #check if file was uploaded
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    
    #Check if the uploaded file has an allowed extension, 
    #save it, extract features, and recommend similar images
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        #extract features from the uploaded image
        features = feature_extraction(os.path.join(app.config['UPLOAD_FOLDER'],filename),model)
        #Find indices of similar images based on extracted features
        indices = recommend(features,feature_list)
        
        images = []
        for index in indices:
            for i in index:
                a = filenames[i].split("\\")
                a = "/".join(a)
                images.append(a) 
        #render the index.html template with the recommended images
        return render_template('index.html', filename=filename, images=images)
    
    else:
        #Flash message if the uploaded file has an invalid extension and redirect to same page
        flash('Allowed image types are - png, jpg, jpeg, jfif, svg')
        return redirect(request.url)

#route to display the uploaded image    
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

#run the flask app
if __name__ == '__main__':
    app.run(debug=True)
    