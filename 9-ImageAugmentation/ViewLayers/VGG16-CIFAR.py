from __future__ import print_function

import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
#%matplotlib inline
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import random
import cv2
from keras.utils import to_categorical




#requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

vgg16_model = keras.applications.vgg16.VGG16()


model_vgg16_custom = Sequential()
for layer in vgg16_model.layers:
    model_vgg16_custom.add(layer)

model_vgg16_custom.layers.pop()

for layer in model_vgg16_custom.layers:
    layer.trainable = False
    
model_vgg16_custom.add(Dense(10, activation='softmax'))


batch_size = 32
#num_classes = 10
#epochs = 100
data_augmentation = True
num_predictions = 20

nb_epoch = 1
nb_classes = 10

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

nb_train_samples = x_train.shape[0]
nb_validation_samples = x_test.shape[0]


# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, nb_classes)
y_test = keras.utils.to_categorical(y_test, nb_classes)
print('y_train shape:', y_train.shape)
print(y_train.shape[0], 'train classes')
print(y_test.shape[0], 'test classes')


# limit the amount of the data
# train data
ind_train = random.sample(list(range(x_train.shape[0])), 10)
x_train = x_train[ind_train]
y_train = y_train[ind_train]


def resize_data(data):
    data_upscaled = np.zeros((data.shape[0], 224, 224, 3))
    for i, img in enumerate(data):
        large_img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        data_upscaled[i] = large_img

    return data_upscaled


# resize train and  test data
x_train_resized = resize_data(x_train)
x_test_resized = resize_data(x_test)

print('x_train_resized shape:', x_train_resized.shape)
print('x_test_resized shape:', x_test_resized.shape)


# make explained variable hot-encoded
y_train_hot_encoded = to_categorical(y_train)
y_test_hot_encoded = to_categorical(y_test)

print('y_train_hot_encoded shape:', y_train_hot_encoded.shape)
print('y_test_hot_encoded shape:', y_test_hot_encoded.shape)

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_datagen.fit(x_train)
train_generator = train_datagen.flow(x_train_resized, y_train, batch_size=32)

test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow(x_test_resized, y_test, batch_size=32)

model_vgg16_custom.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model_vgg16_custom.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)
    #, callbacks=[tb])
    
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('cifar - model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('cifar - model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
