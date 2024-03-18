import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt


# Organize data into train, valid, test directories
os.chdir('Trash-Dataset')   # https://huggingface.co/datasets/garythung/trashnet

if os.path.isdir("train/0/") is False:
    os.mkdir("train")
    os.mkdir("valid")
    os.mkdir("test")

    for i in range(0, 6):
        shutil.move(f'{i}','train')

        os.mkdir(f'valid/{i}')
        os.mkdir(f'test/{i}')

        valid_samples = random.sample(os.listdir(f'train/{i}'), 30)
        for j in valid_samples:
            shutil.move(f'train/{i}/{j}', f'valid/{i}')

        test_samples = random.sample(os.listdir(f'train/{i}'), 5)
        for k in test_samples:
            shutil.move(f'train/{i}/{k}', f'test/{i}')
os.chdir("../..")

# Preprocessing the data
train_path = os.path.join("C:\\Users\\Admin\\Downloads", "Trash-Dataset", "train")
valid_path = os.path.join("C:\\Users\\Admin\\Downloads", "Trash-Dataset", "valid")
test_path = os.path.join("C:\\Users\\Admin\\Downloads", "Trash-Dataset", "test")

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(train_path, target_size=(224,224), batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(valid_path, target_size=(224,224), batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(test_path, target_size=(224,224), batch_size=10, shuffle=False)

# Fine-tuning MobileNet for a custom dataset
mobile = tf.keras.applications.mobilenet.MobileNet()

x = mobile.layers[-5].output
x = tf.keras.layers.Reshape(target_shape=(1024,))(x)
output = Dense(units=6, activation="softmax")(x)

model = Model(inputs=mobile.input, outputs=output)

for layer in model.layers[:-23]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(x=train_batches, validation_data=valid_batches, epochs=30, verbose=2)

model.save("C:\\Users\\Admin\\Downloads\\cnn.h5")

# Testing the model
test_labels = test_batches.classes 
predictions = model.predict(x=test_batches, verbose=0) 

cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))
test_batches.class_indices

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title="Confusion matrix",
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print("Confusion Matrix, without normalization")

    print(cm)

cm_plot_labels = ["0", "1", "2", "3", "4", "5"]
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title="Confusion Matrix")
