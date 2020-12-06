# Import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
# print(tf.__version__)
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
# from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Import the Dataset
df = pd.read_pickle('waferImg26x26.pkl')
images = df.images.values  # independent variable
image_example = np.asarray(images[0])
images_4d = np.zeros((df.shape[0], 26, 26, 3))
for i in range(images_4d.shape[0]):
    images_4d[i, :, :, 0] = images[i][0, :, :]
    images_4d[i, :, :, 1] = images[i][1, :, :]
    images_4d[i, :, :, 2] = images[i][2, :, :]

X = images_4d

# Load the model
cnn = tf.keras.models.load_model('wafer_cnn_128_datagen.h5')

# Making a single prediction
wafer_sample = np.zeros((1, 26, 26, 3))
wafer_sample[:, :, :, :] = X[0, :, :, :]
result = cnn.predict(wafer_sample)

print('done')