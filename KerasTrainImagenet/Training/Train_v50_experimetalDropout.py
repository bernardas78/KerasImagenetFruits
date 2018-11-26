# Trains a model for 4 epochs: 12x12 shifts. Size of image trainable 224x224. with shifts of 12x12, target size is 235
#
# To run:
#   model = t_v50.trainModel(epochs = 20, cntExperiments = 50)

from DataGen import DataGen_v1_150x150_1frame as dg_v1 
from DataGen import AugSequence_v3_randomcrops as as_v3
from Model import Model_v8_sgd as m_v8
import time
from Evaluation import Eval_v1_simple as e_v1
from Evaluation import Eval_v2_top5accuracy as e_v2
from Evaluation import Eval_v3_5framesaccuracy as e_v3
from Evaluation import Eval_v4_10framesaccuracy as e_v4
from keras.callbacks import EarlyStopping
import numpy as np
import os


def trainModel( epochs = 2, cntExperiments = 2 ):
    # Trains a model
    #   model = optional parameter; creates new if not passed; otherwise keeps training
    #   epochs - number of max epochs to train (subject to early stopping)
    #   cntExperiments - count of experiments to run using various random dropout rates
    # Returns: 
    #   model: trained Keras model

    crop_range = 32 # number of pixels to crop image (if size is 235, crops are 0-223, 1-224, ... 11-234)
    target_size = 224
    datasrc = "ilsvrc14_50classes"
    #datasrc = "ilsvrc14"

    #dataGen = dg_v1.prepDataGen(target_size = target_size, batch_size = 64 )
    dataGen = as_v3.AugSequence ( target_size=target_size, crop_range=crop_range, batch_size=128, datasrc=datasrc, test=False, debug=True )

    #prepare a validation data generator, used for early stopping
    vldDataGen = dg_v1.prepDataGen( target_size=target_size, test=True, batch_size=128, datasrc=datasrc )

    input_shape = (target_size, target_size, 3)

    # open a file for writing the results
    res_file = open("train_v50.dropout.results.csv","w") 
    res_file.write ( "Experiment No,cnn1,cnn2,cnn3,cnn4,cnn5,d1,d2,train accuracy,train top5,test accuracy, test top5\n") 

    for exper in range(cntExperiments):

        cnn_dropout = np.array ( [ 0., 0., 0., 0., 0. ])
        incl_cnn_droput = np.random.rand() > 0.0    # 70% chance of using dropout in CNN layers

        # Each dropout value has a 30% chance to be 0; then equal chance of 10%, 20%, ... 70%
        if incl_cnn_droput:
            cnn_dropout = np.maximum (0, np.random.randint (0, 10, 5) - 2 ) * 0.1

        dense_dropout = np.maximum (0, np.random.randint (0, 10, 2) - 2 ) * 0.1

        model = m_v8.prepModel ( input_shape = input_shape, \
            L1_size_stride_filters = (11, 4, 96), L1MaxPool_size_stride = (3, 2), L1_dropout = cnn_dropout[0], \
            L2_size_stride_filters = (5, 1, 256), L2MaxPool_size_stride = (3, 2), L2_dropout = cnn_dropout[1], \
            L3_size_stride_filters = (3, 1, 384),                                 L3_dropout = cnn_dropout[2], \
            L4_size_stride_filters = (3, 1, 384),                                 L4_dropout = cnn_dropout[3], \
            L5_size_stride_filters = (3, 1, 256), L5MaxPool_size_stride = (3, 2), L5_dropout = cnn_dropout[4], \
            D1_size = 4096,                                                       D1_dropout = dense_dropout[0], \
            D2_size = 4096,                                                       D2_dropout = dense_dropout[1], \
            Softmax_size = 50, \
            Conv_padding = "same" )

        #callback_earlystop = EarlyStopping ( monitor='val_acc', min_delta=0.001, patience=20, verbose=1, mode='max', restore_best_weights=True )

        # full epoch is 12x12 = 144 passes over data: 1 times for each subframe
        #model.fit_generator ( dataGen, steps_per_epoch=len(dataGen), epochs=epochs, verbose=2, validation_data=vldDataGen, validation_steps=len(vldDataGen), callbacks=[callback_earlystop] )
        #model.fit_generator ( dataGen, steps_per_epoch=1, epochs=epochs, verbose=2, validation_data=vldDataGen, validation_steps=len(vldDataGen), callbacks=[callback_earlystop] )
        model.fit_generator ( dataGen, steps_per_epoch=len(dataGen), epochs=epochs, verbose=2, validation_data=vldDataGen, validation_steps=len(vldDataGen) )
        #model.fit_generator ( dataGen, steps_per_epoch=len(dataGen), epochs=epochs, verbose=2 )
        #model.fit_generator ( dataGen, steps_per_epoch=1, epochs=epochs, verbose=2 )

        #print ("Evaluation on train set (1 frame)")
        (ev_train, ev_test) = e_v1.eval(model, target_size=target_size, datasrc=datasrc)
     
        res_line = str(exper) + "," + \
            str(cnn_dropout[0]) + "," + \
            str(cnn_dropout[1]) + "," + \
            str(cnn_dropout[2]) + "," + \
            str(cnn_dropout[3]) + "," + \
            str(cnn_dropout[4]) + "," + \
            str(dense_dropout[0]) + "," + \
            str(dense_dropout[1]) + "," + \
            str(ev_train[1]) + "," + \
            str(ev_train[2]) + "," + \
            str(ev_test[1]) + "," + \
            str(ev_test[2]) + "\n"
        res_file.write ( res_line ) 
        res_file.flush()
    
    res_file.close()
    #print ("Evaluation on validation set (1 frame)")
    #e_v2.eval(model, target_size=target_size, datasrc=datasrc, test=True)
    #print ("Evaluation on validation set (5 frames)")
    #e_v3.eval(model, target_size=target_size, datasrc=datasrc, test=True)
    #print ("Evaluation on validation set (10 frames)")
    #e_v4.eval(model, target_size=target_size, datasrc=datasrc, test=True)

    return model