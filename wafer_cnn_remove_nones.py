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

labels = df.labels.values  # dependent variable
labels = np.asarray([str(i[0]) for i in labels])
labels = np.reshape(labels, [-1, 1])  # Re-shape into a column vector

# Remove nones
new_labels = []
new_images = []
for i in range(len(labels)):
    if labels[i] != 'none':
        new_labels.append(labels[i])
        new_images.append(images[i])

labels = np.asarray(new_labels)
images_4d = np.zeros((len(new_images), 26, 26, 3))
for i in range(images_4d.shape[0]):
    images_4d[i, :, :, 0] = new_images[i][0, :, :]
    images_4d[i, :, :, 1] = new_images[i][1, :, :]
    images_4d[i, :, :, 2] = new_images[i][2, :, :]

'''
onehotencoder1 = OneHotEncoder()  # 9 categories, but we only need 8 since one will be all 0s - drop the first column
y = onehotencoder1.fit_transform(labels).toarray()
label_nums = np.sum(y, axis=0)
'''

# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# y = np.array(ct.fit_transform(labels))

# Create Training and Test Dataset + One-hot Encode
X = images_4d

onehotencoder = OneHotEncoder(drop='first')  # 9 categories, but we only need 8 since one will be all 0s - drop the first column
y = onehotencoder.fit_transform(labels).toarray()  # apply the one-hot encoder to the dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)


# Introduce Transformations for Image Augmentation to reduce overfitting - Vertical and Horizontal image flipping
datagen = ImageDataGenerator(
    vertical_flip=True,
    horizontal_flip=True)
datagen.fit(X_train)

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[26, 26, 3]))
# Max Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Second Convolution Layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flattening
cnn.add(tf.keras.layers.Flatten())

# Full Connection
cnn.add(tf.keras.layers.Dense(units=64, activation='relu'))

# Output Layer
cnn.add(tf.keras.layers.Dense(units=7, activation='sigmoid'))

# Compiling the CNN
opt = tf.keras.optimizers.Adam(learning_rate=0.01)  # Modify the learning rate of the Adam optimizer used
'''
# If you want to use a learning rate that changes over time - Doesn't have a large effect...
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
'''
cnn.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

print(cnn.summary())  # Print CNN architecture

# history = cnn.fit(X_train, y_train, batch_size=32, epochs=25, validation_data=(X_test, y_test))  # Fit the CNN to the training data
# cnn.save('wafer_cnn_128.h5')

history = cnn.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=100, validation_data=(X_test, y_test))  # Fit the CNN to the augmented training data
cnn.save('wafer_cnn_removed_nones_128_datagen.h5')  # Save the model

# Plot Accuracy vs Epochs for Training and Test set, with legends
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()

print('done')

