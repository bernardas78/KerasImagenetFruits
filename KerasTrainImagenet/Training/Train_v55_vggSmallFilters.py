# Trains a model for 4 epochs: 12x12 shifts. Size of image trainable 224x224. with shifts of 12x12, target size is 235
#
# To run:
#   model = t_v55.trainModel(epochs=200)

#from DataGen import AugSequence_v3_randomcrops as as_v3
from DataGen import AugSequence_v4_PcaDistortion as as_v4
from Model import Model_v10_vgg as m_v10
import time
from Evaluation import Eval_v2_top5accuracy as e_v2
from Evaluation import Eval_v3_5framesaccuracy as e_v3
from Evaluation import Eval_v4_10framesaccuracy as e_v4
from keras.callbacks import EarlyStopping
import numpy as np


def trainModel( model = None, epochs = 1):
    # Trains a model
    #   model = optional parameter; creates new if not passed; otherwise keeps training
    #   epochs - number of max epochs to train (subject to early stopping)
    # Returns: 
    #   model: trained Keras model

    crop_range = 32 # number of pixels to crop image (if size is 235, crops are 0-223, 1-224, ... 11-234)
    target_size = 224
    datasrc = "ilsvrc14_100classes"
    #datasrc = "ilsvrc14"

    # "presumed mean" of X, subtract from all input
    #subtractMean=0.5

    # Load pre-calculated RGB mean, PCA (Principal Component Analysis) eigenvectors and eigenvalues
    #subtractMean=np.array ( [ 0.4493, 0.4542, 0.3901 ] )
    subtractMean = np.load("..\\rgb_mean.npy")
    pca_eigenvectors = np.load("..\\eigenvectors.npy")
    pca_eigenvalues = np.load("..\\eigenvalues.npy")

    #dataGen = dg_v1.prepDataGen(target_size = target_size, batch_size = 64 )
    dataGen = as_v4.AugSequence ( target_size=target_size, crop_range=crop_range, allow_hor_flip=True, batch_size=48, \
        subtractMean=subtractMean, pca_eigenvectors=pca_eigenvectors, pca_eigenvalues=pca_eigenvalues, \
        datasrc=datasrc, test=False )

    if model is None:
        input_shape = (target_size, target_size, 3)
        model = m_v10.prepModel ( input_shape = input_shape, \
            L1_size_stride_filters = (3, 1, 64),  L1MaxPool_size_stride = (2, 2), L1_dropout = 0.0, \
            L2_size_stride_filters = (3, 1, 128), L2MaxPool_size_stride = (2, 2), L2_dropout = 0.0, \
            L3_size_stride_filters = (3, 1, 256),                                 L3_dropout = 0.0, \
            L4_size_stride_filters = (3, 1, 256), L4MaxPool_size_stride = (2, 2), L4_dropout = 0.0, \
            L5_size_stride_filters = (3, 1, 512),                                 L5_dropout = 0.0, \
            L6_size_stride_filters = (3, 1, 512), L6MaxPool_size_stride = (2, 2), L6_dropout = 0.0, \
            L7_size_stride_filters = (3, 1, 512),                                 L7_dropout = 0.0, \
            L8_size_stride_filters = (3, 1, 512), L8MaxPool_size_stride = (2, 2), L8_dropout = 0.0, \
            D1_size = 4096,                                                       D1_dropout = 0.5, \
            D2_size = 4096,                                                       D2_dropout = 0.5, \
            Softmax_size = 100, \
            Conv_padding = "same" )

    #prepare a validation data generator, used for early stopping
    #vldDataGen = dg_v1.prepDataGen( target_size=target_size, test=True, batch_size=128, datasrc=datasrc )
    vldDataGen = as_v4.AugSequence ( target_size=target_size, crop_range=1, allow_hor_flip=False, batch_size=48, subtractMean=subtractMean, datasrc=datasrc, test=True )
    callback_earlystop = EarlyStopping ( monitor='val_acc', min_delta=0.001, patience=20, verbose=1, mode='max', restore_best_weights=True )

    # full epoch is 12x12 = 144 passes over data: 1 times for each subframe
    model.fit_generator ( dataGen, steps_per_epoch=len(dataGen), epochs=epochs, verbose=2, validation_data=vldDataGen, validation_steps=len(vldDataGen), callbacks=[callback_earlystop] )
    #model.fit_generator ( dataGen, steps_per_epoch=1, epochs=epochs, verbose=2, validation_data=vldDataGen, validation_steps=len(vldDataGen), callbacks=[callback_earlystop] )
    #model.fit_generator ( dataGen, steps_per_epoch=len(dataGen), epochs=epochs, verbose=2 )
    #model.fit_generator ( dataGen, steps_per_epoch=1, epochs=epochs, verbose=2 )

    print ("Evaluation on train set (1 frame)")
    e_v2.eval(model, target_size=target_size, subtractMean=subtractMean, datasrc=datasrc)
    print ("Evaluation on validation set (1 frame)")
    e_v2.eval(model, target_size=target_size, subtractMean=subtractMean, datasrc=datasrc, test=True)
    print ("Evaluation on validation set (5 frames)")
    e_v3.eval(model, target_size=target_size, subtractMean=subtractMean, datasrc=datasrc, test=True)
    print ("Evaluation on validation set (10 frames)")
    e_v4.eval(model, target_size=target_size, subtractMean=subtractMean, datasrc=datasrc, test=True)

    return model