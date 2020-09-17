# importing the necessary packages and libraries

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow.keras as keras
import tensorflow as tf
from tqdm import tqdm

training_path = "mcdonalds/train"
testing_path = "mcdonalds/test"
categories = ['big_mac_burger', 'fries', 'coke']

# To read and display the image in graysclae 
# path = "mcdonalds/train/fries/00000000.jpg"
# img_array = cv2.imread(path ,cv2.IMREAD_GRAYSCALE)  # convert to array
# plt.imshow(img_array, cmap='gray')  # graph it
# plt.show()
# print(img_array)
# print(img_array.shape)

#resizing the image and coverting to array of pixels
IMG_SIZE = 500
training_data = []
for category in categories:
    path = os.path.join(training_path,category)
    class_num = categories.index(category)
    for img in tqdm(os.listdir(path)):
        # print(img)
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        training_data.append([new_array, class_num])

print(len(training_data))

#shuffling the data 
random.shuffle(training_data)
X_train = []
y_train = []
for features,label in training_data:
    X_train.append(features)
    y_train.append(label)
X_train = tf.keras.utils.normalize(X_train, axis=1)
print(type(y_train))
y_train = np.array(y_train)

#building and training the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=4)

# saving or pickling the trained model
model.save('mcd_classifier_model.model')
