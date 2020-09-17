import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow.keras as keras
import tensorflow as tf
from tqdm import tqdm
import numpy as np

# loading or unpickling the trained model
new_model = tf.keras.models.load_model('mcd_classifier.model')

#predicting with a test image
IMG_SIZE = 500
img = "mcdonalds/test/t2.jpeg"
img_array = cv2.imread(img ,cv2.IMREAD_GRAYSCALE)
test_img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
x_test = tf.keras.utils.normalize([test_img], axis=1)

predictions = new_model.predict(x_test)
print('These are the predictions:',predictions)


print(np.argmax(predictions[0]))
