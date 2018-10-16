# Prepares a simple model
#   Sample downloaded from https://elitedatascience.com/keras-tutorial-deep-learning-in-python
#
# To run: 
#   model = m_v2.prepModel()

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout

def prepModel( input_shape=(150,150,3), \
    L1_size_stride_filters = (3, 1, 32), L1MaxPool_size_stride = None, \
    L2_size_stride_filters = (3, 1, 32), L2MaxPool_size_stride = (2, 1) \
   ):

    L1_size = L1_size_stride_filters[0]
    L1_stride = L1_size_stride_filters[1]
    L1_filters = L1_size_stride_filters[2]

    L2_size = L2_size_stride_filters[0]
    L2_stride = L2_size_stride_filters[1]
    L2_filters = L2_size_stride_filters[2]

    model = Sequential()
 
    model.add( Convolution2D ( filters = L1_filters,  kernel_size = (L1_size, L1_size),  strides = (L1_stride, L1_stride),  activation='relu',  input_shape=input_shape) )
    if L1MaxPool_size_stride is not None:
        L1MaxPool_size = L1MaxPool_size_stride[0]
        L1MaxPool_stride = L1MaxPool_size_stride[1]
        model.add ( MaxPooling2D ( pool_size = ( L1MaxPool_size , L1MaxPool_size ), strides = ( L1MaxPool_stride, L1MaxPool_stride ) ) )

    model.add( Convolution2D ( filters = L2_filters,  kernel_size = (L2_size, L2_size),  strides = (L2_stride, L2_stride),  activation='relu' ) )
    if L2MaxPool_size_stride is not None:
        L2MaxPool_size = L2MaxPool_size_stride[0]
        L2MaxPool_stride = L2MaxPool_size_stride[1]
        model.add ( MaxPooling2D ( pool_size = ( L2MaxPool_size , L2MaxPool_size ), strides = ( L2MaxPool_stride, L2MaxPool_stride ) ) )
        #model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    return model