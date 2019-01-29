# Trains a model for 4 epochs: 12x12 shifts. Size of image trainable 224x224. with shifts of 12x12, target size is 235
#
# To run:
#   model = t_det_v62.trainModel(epochs=10)

#from DataGen import AugSequence_v3_randomcrops as as_v3
#from DataGen import AugSequence_v4_PcaDistortion as as_v4
from DataGen_DET import AugSequence_v7_det_simplest as as_det_v7
#from Model import Model_v8_sgd as m_v8
from Model_DET import Model_v13_det_simplest as m_det_v13
#from Model import Model_v8_sgd as m_v8
#import time
from Evaluation import Eval_v2_top5accuracy as e_v2
from Evaluation import Eval_v3_5framesaccuracy as e_v3
from Evaluation import Eval_v4_10framesaccuracy as e_v4
from keras.callbacks import EarlyStopping
import numpy as np

def trainModel( epochs = 1):
    # Trains a model
    #   model = optional parameter; creates new if not passed; otherwise keeps training
    #   epochs - number of max epochs to train (subject to early stopping)
    # Returns: 
    #   model: trained Keras model

    crop_range = 32 # number of pixels to crop image (if size is 235, crops are 0-223, 1-224, ... 11-234)
    target_size = 224
    datasrc = "ilsvrc14_DET"
    #datasrc = "ilsvrc14"

    # "presumed mean" of X, subtract from all input
    #subtractMean=0.5

    # Load pre-calculated RGB mean, PCA (Principal Component Analysis) eigenvectors and eigenvalues
    #subtractMean=np.array ( [ 0.4493, 0.4542, 0.3901 ] )
    #subtractMean = np.load("..\\rgb_mean.npy")
    #pca_eigenvectors = np.load("..\\eigenvectors.npy")
    #pca_eigenvalues = np.load("..\\eigenvalues.npy")

    #dataGen = dg_v1.prepDataGen(target_size = target_size, batch_size = 64 )
    dataGen = as_det_v7.AugSequence ( target_size=target_size, \
        #crop_range=crop_range, allow_hor_flip=True, 
        batch_size=64, 
        #subtractMean=subtractMean, pca_eigenvectors=pca_eigenvectors, pca_eigenvalues=pca_eigenvalues, \
        #preprocess="vgg",
        datasrc=datasrc, test=False, debug=True )

    model = m_det_v13.prepModel ( target_size=target_size )

    #prepare a validation data generator, used for early stopping
    #vldDataGen = dg_v1.prepDataGen( target_size=target_size, test=True, batch_size=128, datasrc=datasrc )
    vldDataGen = as_det_v7.AugSequence ( target_size=target_size,\
        #crop_range=1, 
        batch_size=64, \
        #subtractMean=subtractMean, 
        #preprocess="vgg", 
        datasrc=datasrc, test=True, debug=True )
    #callback_earlystop = EarlyStopping ( monitor='val_acc', min_delta=0.001, patience=20, verbose=1, mode='max', restore_best_weights=True )

    # full epoch is 12x12 = 144 passes over data: 1 times for each subframe
    model.fit_generator ( dataGen, steps_per_epoch=len(dataGen), epochs=epochs, verbose=2, validation_data=vldDataGen \
        , validation_steps=len(vldDataGen) \
        #, callbacks=[callback_earlystop] \
        )

    #print ("Evaluation on train set (1 frame)")
    #e_v2.eval(model, target_size=target_size,  datasrc=datasrc)
    #print ("Evaluation on validation set (1 frame)")
    #e_v2.eval(model, target_size=target_size, datasrc=datasrc, preprocess="vgg", test=True)
    #print ("Evaluation on validation set (5 frames)")
    #e_v3.eval(model, target_size=target_size, datasrc=datasrc, preprocess="vgg", test=True)
    #print ("Evaluation on validation set (10 frames)")
    #e_v4.eval(model, target_size=target_size, datasrc=datasrc, preprocess="vgg", test=True)

    return model