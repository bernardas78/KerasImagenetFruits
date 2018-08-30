# Prepares a simple model
#   Sample downloaded from https://elitedatascience.com/keras-tutorial-deep-learning-in-python
#
# To run: 
#   model = m_v2.prepModel()

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout

def prepModel():
    model = Sequential()
 
    model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(150,150,3)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    return model