# Prepares a simple model
#   Sample downloaded from https://elitedatascience.com/keras-tutorial-deep-learning-in-python
#
# To run: 
#   model = m_v13.prepModel()

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from keras import regularizers
import numpy as np
from keras.optimizers import Adam

def prepModel( target_size, Softmax_size ) :

    input_shape = (target_size,target_size,3)
    bn_layers= ["c+1", "c+2", "c+3", "c+4", "c+5", "c+6", "c+7", "c+8", "d-2", "d-3"]
    dropout_layers= ["d-2", "d-3"]
    l2_layers= {}
    padding= "same"
    dense_sizes= {"d-3": 256, "d-2": 128, "d-1": Softmax_size}
    conv_layers_over_5= 2
    use_maxpool_after_conv_layers_after_5th= [False, True]

    model = Sequential()

    # 1st CNN
    model.add(Convolution2D(32, (3,3), input_shape=input_shape, padding=padding))
    if "c+1" in bn_layers:
        model.add (BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 2nd CNN
    model.add(Convolution2D(64, (3,3), padding=padding))
    if "c+2" in bn_layers:
        model.add (BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 3rd CNN
    model.add(Convolution2D(128, (3, 3), padding=padding))
    if "c+3" in bn_layers:
        model.add (BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 4th CNN
    model.add(Convolution2D(256, (3, 3), padding=padding))
    if "c+4" in bn_layers:
        model.add (BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 5th CNN
    model.add(Convolution2D(256, (3, 3), padding=padding))
    if "c+5" in bn_layers:
        model.add (BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 6th+ CNN
    for layer_ind_after_5th in np.arange ( conv_layers_over_5 ):
        model.add(Convolution2D(256, (3, 3), padding=padding))
        if "c+"+str(layer_ind_after_5th+6) in bn_layers:
            model.add (BatchNormalization())
        model.add(Activation('relu'))
        if use_maxpool_after_conv_layers_after_5th[layer_ind_after_5th]:
            model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # -3rd dense
    d_3_regularizer = regularizers.l2( l2_layers["d-3"] ) if "d-3" in l2_layers else None
    d_3_size = dense_sizes["d-3"]
    model.add(Dense(d_3_size, kernel_regularizer=d_3_regularizer, bias_regularizer=d_3_regularizer))
    if "d-3" in bn_layers:
        model.add (BatchNormalization())
    if "d-3" in dropout_layers:
        model.add (Dropout(rate=0.5))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    # -2nd dense
    #model.add(Dense(128, activation='relu'))
    d_2_regularizer = regularizers.l2( l2_layers["d-2"] ) if "d-2" in l2_layers else None
    d_2_size = dense_sizes["d-2"]
    model.add(Dense(d_2_size, kernel_regularizer=d_2_regularizer, bias_regularizer=d_2_regularizer))
    if "d-2" in bn_layers:
        model.add (BatchNormalization())
    if "d-2" in dropout_layers:
        model.add (Dropout(rate=0.5))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    # -1st dense
    d_1_size = dense_sizes["d-1"] if "d-1" in dense_sizes else 6
    model.add(Dense(d_1_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001), #'adam', # default LR: 0.001
                  metrics=['accuracy'])

    return model