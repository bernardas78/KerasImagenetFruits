# Prepares a simple model
#   Sample downloaded from https://elitedatascience.com/keras-tutorial-deep-learning-in-python
#
# To run: 
#   model = m_v7.prepModel()

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout

def prepModel( input_shape=(150,150,3), \
    L1_size_stride_filters = (3, 1, 32), L1MaxPool_size_stride = None, \
    L2_size_stride_filters = (3, 1, 32), L2MaxPool_size_stride = (2, 1), \
    L3_size_stride_filters = (3, 1, 384), \
    L4_size_stride_filters = (3, 1, 384),                                 L4_dropout = 0.25, \
    L5_size_stride_filters = (3, 1, 256), L5MaxPool_size_stride = (3, 2), \
    D1_size = 128, \
    D2_size = None, \
    Softmax_size = 20, \
    Conv_padding = "valid" ) :

    L1_size = L1_size_stride_filters[0]
    L1_stride = L1_size_stride_filters[1]
    L1_filters = L1_size_stride_filters[2]

    L2_size = L2_size_stride_filters[0]
    L2_stride = L2_size_stride_filters[1]
    L2_filters = L2_size_stride_filters[2]

    L3_size = L3_size_stride_filters[0]
    L3_stride = L3_size_stride_filters[1]
    L3_filters = L3_size_stride_filters[2]

    L4_size = L4_size_stride_filters[0]
    L4_stride = L4_size_stride_filters[1]
    L4_filters = L4_size_stride_filters[2]

    L5_size = L5_size_stride_filters[0]
    L5_stride = L5_size_stride_filters[1]
    L5_filters = L5_size_stride_filters[2]

    model = Sequential()
 
    model.add( Convolution2D ( filters = L1_filters,  kernel_size = (L1_size, L1_size),  strides = (L1_stride, L1_stride), padding = Conv_padding, activation='relu',  input_shape=input_shape) )
    if L1MaxPool_size_stride is not None:
        L1MaxPool_size = L1MaxPool_size_stride[0]
        L1MaxPool_stride = L1MaxPool_size_stride[1]
        model.add ( MaxPooling2D ( pool_size = ( L1MaxPool_size , L1MaxPool_size ), strides = ( L1MaxPool_stride, L1MaxPool_stride ) ) )

    model.add( Convolution2D ( filters = L2_filters,  kernel_size = (L2_size, L2_size),  strides = (L2_stride, L2_stride), padding = Conv_padding, activation='relu' ) )
    if L2MaxPool_size_stride is not None:
        L2MaxPool_size = L2MaxPool_size_stride[0]
        L2MaxPool_stride = L2MaxPool_size_stride[1]
        model.add ( MaxPooling2D ( pool_size = ( L2MaxPool_size , L2MaxPool_size ), strides = ( L2MaxPool_stride, L2MaxPool_stride ) ) )
        #model.add(MaxPooling2D(pool_size=(2,2)))

    model.add( Convolution2D ( filters = L3_filters,  kernel_size = (L3_size, L3_size),  strides = (L3_stride, L3_stride), padding = Conv_padding,  activation='relu' ) )

    model.add( Convolution2D ( filters = L4_filters,  kernel_size = (L4_size, L4_size),  strides = (L4_stride, L4_stride), padding = Conv_padding,  activation='relu' ) )
    if L4_dropout > 0.:
        model.add(Dropout(L4_dropout))

    model.add( Convolution2D ( filters = L5_filters,  kernel_size = (L5_size, L5_size),  strides = (L5_stride, L5_stride), padding = Conv_padding,  activation='relu' ) )
    if L5MaxPool_size_stride is not None:
        L5MaxPool_size = L5MaxPool_size_stride[0]
        L5MaxPool_stride = L5MaxPool_size_stride[1]
        model.add ( MaxPooling2D ( pool_size = ( L5MaxPool_size , L5MaxPool_size ), strides = ( L5MaxPool_stride, L5MaxPool_stride ) ) )

    model.add(Flatten())
    
    model.add(Dense(D1_size, activation='relu'))
    model.add(Dropout(0.5))
    
    if D2_size is not None:
        model.add(Dense(D2_size, activation='relu'))
        model.add(Dropout(0.5))

    model.add ( Dense ( Softmax_size, activation='softmax' ) )

    model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

    return model