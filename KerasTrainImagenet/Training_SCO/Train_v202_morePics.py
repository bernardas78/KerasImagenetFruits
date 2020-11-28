# Trains a model for 4 epochs: 12x12 shifts. Size of image trainable 224x224. with shifts of 12x12, target size is 235
#
# To run:
#   model = t_sco_v202.trainModel(epochs=200)

#from DataGen import AugSequence_v3_randomcrops as as_v3
#from DataGen import AugSequence_v4_PcaDistortion as as_v4
from DataGen import AugSequence_v5_vggPreprocess as as_v5
#from Model import Model_v8_sgd as m_v8
from Model import Model_v13_Visible as m_v13
#from Model import Model_v11_pretVggPlusSoftmax as m_v11
#from Model import Model_v7_5cnn as m_v7
from keras.models import load_model
#import time
from Evaluation import Eval_v2_top5accuracy as e_v2
from Evaluation import Eval_v3_5framesaccuracy as e_v3
from Evaluation import Eval_v4_10framesaccuracy as e_v4
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from datetime import date

# model file
model_temp_file_pattern = "A:\\RetellectModels\\model_{}_{}prekes.h5"

def trainModel( epochs = 1):
    # Trains a model
    #   model = optional parameter; creates new if not passed; otherwise keeps training
    #   epochs - number of max epochs to train (subject to early stopping)
    # Returns: 
    #   model: trained Keras model

    crop_range = 1 # number of pixels to crop image (if size is 235, crops are 0-223, 1-224, ... 11-234)
    target_size = 256
    datasrc = "sco_v3"
    #Softmax_size=33
    
    # "presumed mean" of X, subtract from all input
    #subtractMean=0.5

    # Load pre-calculated RGB mean, PCA (Principal Component Analysis) eigenvectors and eigenvalues
    #subtractMean=np.array ( [ 0.4493, 0.4542, 0.3901 ] )
    #subtractMean = np.load("..\\rgb_mean.npy")
    #pca_eigenvectors = np.load("..\\eigenvectors.npy")
    #pca_eigenvalues = np.load("..\\eigenvalues.npy")

    #dataGen = dg_v1.prepDataGen(target_size = target_size, batch_size = 64 )
    dataGen = as_v5.AugSequence ( target_size=target_size, crop_range=crop_range, allow_hor_flip=True, batch_size=32, \
        #subtractMean=subtractMean, pca_eigenvectors=pca_eigenvectors, pca_eigenvalues=pca_eigenvalues, \
        preprocess="div255", datasrc=datasrc, test=False, debug=True )

    Softmax_size=len(dataGen.dataGen().class_indices)

    model = m_v13.prepModel (Softmax_size=Softmax_size, L1MaxPool_size_stride=(2,2), L2MaxPool_size_stride=(2,2), L5MaxPool_size_stride=(2,2) )
    model.summary()

    #prepare a validation data generator, used for early stopping
    vldDataGen = as_v5.AugSequence ( target_size=target_size, crop_range=1, allow_hor_flip=False, batch_size=32, \
        preprocess="div255", datasrc=datasrc, test=True )
    callback_earlystop = EarlyStopping ( monitor='val_acc', min_delta=0.001, patience=20, verbose=1, mode='max')#, restore_best_weights=True )
    #callback_earlystop = EarlyStopping ( monitor='val_acc', min_delta=0., patience=0, verbose=2, mode='auto', restore_best_weights=True )
    model_temp_file = model_temp_file_pattern.format( date.today().strftime("%Y%m%d"), Softmax_size )
    mcp_save = ModelCheckpoint(model_temp_file, save_best_only=True, monitor='val_acc', mode='max')

    # full epoch is 12x12 = 144 passes over data: 1 times for each subframe
    model.fit_generator ( dataGen, steps_per_epoch=len(dataGen), epochs=epochs, verbose=2, 
                         validation_data=vldDataGen, validation_steps=len(vldDataGen), callbacks=[callback_earlystop,mcp_save] )
    #model.fit_generator ( dataGen, steps_per_epoch=len(dataGen), epochs=epochs, verbose=2 )

    # Loading best saved model
    model = load_model(model_temp_file)

    #print ("Evaluation on train set (1 frame)")
    #e_v2.eval(model, target_size=target_size,  datasrc=datasrc)
    print ("Evaluation on validation set (1 frame)")
    e_v2.eval(model, target_size=target_size, datasrc=datasrc, preprocess="div255", test=True)
    print ("Evaluation on validation set (5 frames)")
    e_v3.eval(model, target_size=target_size, datasrc=datasrc, preprocess="div255", test=True)
    print ("Evaluation on validation set (10 frames)")
    e_v4.eval(model, target_size=target_size, datasrc=datasrc, preprocess="div255", test=True)

    #print ("Evaluation on test set (1 frame)")
    e_v2.eval(model, target_size=target_size, datasrc=datasrc, preprocess="div255", testtest=True)

    return model