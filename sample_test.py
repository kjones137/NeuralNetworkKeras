#!/usr/bin/env python3

#file to test model on a small array of inputs

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from digit_classifier import x_test, test_labels
import numpy as np 

#build model as structured in digit_classifier.py

model = Sequential([
    Dense(64, activation = 'relu', input_shape = (784,)),
    Dense(64, activation = 'relu'),
    Dense(10, activation = 'softmax'),
])

#load weights stored from digit_classifier.py

model.load_weights('model.weights.h5')

#generate predictions for 5 test images

preds = model.predict(x_test[95:100])

#view predictions

print(np.argmax(preds, axis = 1))

#view actual values of input images

print(test_labels[95:100])

