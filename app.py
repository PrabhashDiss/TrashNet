from flask import Flask, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.optimizers import Adam
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)

# Create the "uploads" directory if it doesn't exist
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Load the trained model
model = load_model("C:\\Users\\Admin\\Downloads\\cnn.h5")

# Mapping of class indices to class labels
class_labels = {0: "Trash", 1: "Cardboard", 2: "Glass", 3: "Metal", 4: "Paper", 5: "Plastic"}

def predict_image(img_path):
    # Load the image and resize it to the target size expected by the model (224x224)
    img = load_img(img_path, target_size=(224, 224))
    
    # Convert the image to a NumPy array and expand its dimensions to create a batch of one image
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the pixel values of the image array to the range [0, 1]
    #img_array = img_array / 255.0
    img_preprocessed = tf.keras.applications.mobilenet.preprocess_input(img_array)  # Preprocess the image
    
    # Make a prediction using the trained model
    prediction = model.predict(img_preprocessed)
    print("Prediction:", prediction)
    
    # Get the index of the predicted class (the class with the highest probability)
    predicted_class = np.argmax(prediction)
    print("Predicted class:", predicted_class)

    # Map the predicted class index to the corresponding class label
    if predicted_class in class_labels:
        predicted_label = class_labels[predicted_class]
    else:
        predicted_label = "Unknown Class"
    
    return predicted_label

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Predict the image
    prediction = predict_image(file_path)

    # Render the template with the prediction result
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prediction Result</title>
    </head>
    <body>
        <h1>Prediction Result</h1>
        <p>The predicted class is: { prediction }</p>
    </body>
    </html>
    '''

@app.route('/')
def upload_form():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload new Image</title>
    </head>
    <body>
        <h1>Upload new Image</h1>
        <form method="post" enctype="multipart/form-data" action="/upload">
            <input type="file" name="file">
            <input type="submit" value="Upload">
        </form>
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(debug=True)
