# Trains a model for 4 epochs: 12x12 shifts. Size of image trainable 224x224. with shifts of 12x12, target size is 235
#
# To run:
#   model = t_v58.trainModel(epochs=200)

#from DataGen import AugSequence_v3_randomcrops as as_v3
#from DataGen import AugSequence_v4_PcaDistortion as as_v4
from DataGen import AugSequence_v5_vggPreprocess as as_v5
from Model import Model_v8_sgd as m_v8
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
    datasrc = "ilsvrc14_100boundingBoxes"
    #datasrc = "ilsvrc14"

    # "presumed mean" of X, subtract from all input
    #subtractMean=0.5

    # Load pre-calculated RGB mean, PCA (Principal Component Analysis) eigenvectors and eigenvalues
    #subtractMean = np.load("..\\rgb_mean.npy")
    subtractMean = np.array([123.68, 116.779, 103.939])/255. # actual mean of entire 1000 classes
    #pca_eigenvectors = np.load("..\\eigenvectors.npy")
    #pca_eigenvalues = np.load("..\\eigenvalues.npy")

    #dataGen = dg_v1.prepDataGen(target_size = target_size, batch_size = 64 )
    dataGen = as_v5.AugSequence ( target_size=target_size, crop_range=crop_range, allow_hor_flip=True, batch_size=128, \
        subtractMean=subtractMean, \
        #pca_eigenvectors=pca_eigenvectors, pca_eigenvalues=pca_eigenvalues, 
        preprocess="div255",\
        datasrc=datasrc, test=False )

    if model is None:
        input_shape = (target_size, target_size, 3)
        model = m_v8.prepModel ( input_shape = input_shape, \
            L1_size_stride_filters = (11, 4, 96), L1MaxPool_size_stride = (3, 2), L1_dropout = 0.0, \
            L2_size_stride_filters = (5, 1, 256), L2MaxPool_size_stride = (3, 2), L2_dropout = 0.0, \
            L3_size_stride_filters = (3, 1, 384),                                 L3_dropout = 0.0, \
            L4_size_stride_filters = (3, 1, 384),                                 L4_dropout = 0.0, \
            L5_size_stride_filters = (3, 1, 256), L5MaxPool_size_stride = (3, 2), L5_dropout = 0.0, \
            D1_size = 4096,                                                       D1_dropout = 0.5, \
            D2_size = 4096,                                                       D2_dropout = 0.55, \
            Softmax_size = 100, \
            Conv_padding = "same" )

    #prepare a validation data generator, used for early stopping
    #vldDataGen = dg_v1.prepDataGen( target_size=target_size, test=True, batch_size=128, datasrc=datasrc )
    vldDataGen = as_v5.AugSequence ( target_size=target_size, crop_range=1, allow_hor_flip=False, batch_size=128, \
        subtractMean=subtractMean, 
        preprocess="div255", datasrc=datasrc, test=True )
    callback_earlystop = EarlyStopping ( monitor='val_acc', min_delta=0.001, patience=20, verbose=1, mode='max', restore_best_weights=True )

    # full epoch is 12x12 = 144 passes over data: 1 times for each subframe
    model.fit_generator ( dataGen, steps_per_epoch=len(dataGen), epochs=epochs, verbose=2, validation_data=vldDataGen, validation_steps=len(vldDataGen), callbacks=[callback_earlystop] )
    #model.fit_generator ( dataGen, steps_per_epoch=1, epochs=epochs, verbose=2, validation_data=vldDataGen, validation_steps=len(vldDataGen), callbacks=[callback_earlystop] )
    #model.fit_generator ( dataGen, steps_per_epoch=len(dataGen), epochs=epochs, verbose=2 )
    #model.fit_generator ( dataGen, steps_per_epoch=1, epochs=epochs, verbose=2 )

    print ("Evaluation on train set (1 frame)")
    e_v2.eval(model, target_size=target_size, subtractMean=subtractMean, datasrc=datasrc, preprocess="div255" )
    print ("Evaluation on validation set (1 frame)")
    e_v2.eval(model, target_size=target_size, subtractMean=subtractMean, datasrc=datasrc, preprocess="div255", test=True)
    print ("Evaluation on validation set (5 frames)")
    e_v3.eval(model, target_size=target_size, subtractMean=subtractMean, datasrc=datasrc, preprocess="div255", test=True)
    print ("Evaluation on validation set (10 frames)")
    e_v4.eval(model, target_size=target_size, subtractMean=subtractMean, datasrc=datasrc, preprocess="div255", test=True)

    return model