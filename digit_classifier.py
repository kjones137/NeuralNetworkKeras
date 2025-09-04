#!/usr/bin/env python3

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

x_train, x_test = np.array(train_images, np.float32), np.array(test_images, np.float32)

#Normalize images to transform pixel values from [0, 255] to [-.5, .5] to train

x_train = (x_train / 255) - .5
x_test = (x_test / 255) - .5

#Flatten images

x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))


#print(x_train.shape)
#print(x_test.shape)


#define model

model = Sequential([
    Dense(64, activation = 'relu'),
    Dense(64, activation = 'relu'),
    Dense(10, activation = 'softmax'),
])


#compile model with optimizer, loss function, and metrics

model.compile(
        optimizer = 'adam',
        loss = 'categorical_crossentropy',
        metrics = ['accuracy'],
)


#training the model

model.fit(
        x_train,
        to_categorical(train_labels),
        epochs = 5,
        batch_size = 32,
)


#evaluate model using test data

model.evaluate (
        x_test,
        to_categorical(test_labels)
)

model.save_weights('model.weights.h5')
