# Trains a model for 4 epochs: 12x12 shifts. Size of image trainable 224x224. with shifts of 12x12, target size is 235
#
# To run:
#   from Training import Train_debugFit as t_debug
#   importlib.reload(t_debug)
#   model = t_debug.trainModel()

from DataGen import DataGen_v1_150x150_1frame as dg_v1 
from DataGen import AugSequence as as_v1
from Model import Model_v7_5cnn as m_v7
import time
from Evaluation import Eval_v1_simple as e_v1
from Evaluation import Eval_v2_top5accuracy as e_v2


def trainModel( model = None):
    # Trains a model
    #   model = optional parameter; creates new if not passed; otherwise keeps training
    # Returns: 
    #   model: trained Keras model

    crop_range = 12 # number of pixels to crop image (if size is 235, crops are 0-223, 1-224, ... 11-234)
    crop_size = 224
    target_size = crop_size + crop_range - 1 #235
    input_shape = (224, 224, 3)

    ################### FIT GENERATOR #################
    #dataGen = dg_v1.prepDataGen(target_size = 224, batch_size = 64 )
    dataGen = as_v1.AugSequence ( target_size=224, crop_range=2, batch_size=256, test=False )

    model = m_v7.prepModel ( input_shape = input_shape, \
        L1_size_stride_filters = (7, 2, 96), L1MaxPool_size_stride = (3, 2), \
        L2_size_stride_filters = (5, 2, 256), L2MaxPool_size_stride = (3, 2), \
        L3_size_stride_filters = (3, 1, 384), \
        L4_size_stride_filters = (3, 1, 384), \
        L5_size_stride_filters = (3, 1, 256), L5MaxPool_size_stride = (3, 2), \
        D1_size = 4096, \
        D2_size = 128)

    print ("FIT GENERATOR START ", time.strftime("%H:%M:%S"))
    model.fit_generator ( dataGen, steps_per_epoch=len(dataGen), epochs=2, verbose=2 )
    print ("FIT GENERATOR END ", time.strftime("%H:%M:%S"))
    e_v2.eval(model, target_size=crop_size)
    e_v2.eval(model, target_size=crop_size, test=True)

    #################### FIT LOOP #################
    #model = m_v7.prepModel ( input_shape = input_shape, \
    #    L1_size_stride_filters = (7, 2, 96), L1MaxPool_size_stride = (3, 2), \
    #    L2_size_stride_filters = (5, 2, 256), L2MaxPool_size_stride = (3, 2), \
    #    L3_size_stride_filters = (3, 1, 384), \
    #    L4_size_stride_filters = (3, 1, 384), \
    #    L5_size_stride_filters = (3, 1, 256), L5MaxPool_size_stride = (3, 2), \
    #    D1_size = 4096, \
    #    D2_size = 128)

    ##dataGen = dg_v1.prepDataGen(target_size = 224, batch_size = 64 )
    #dataGen = as_v1.AugSequence ( target_size=224, batch_size=64, test=False )

    ## full epoch is 12x12 = 144 passes over data: 1 times for each subframe
    ##print (time.strftime("%H:%M:%S"))
    #for full_epoch in range (20):

    #    # for each subframe within a image...
    #    for epoch_single_subframe in range ( 1 * 1 ):

    #        #Pick a starting pixel 
    #        h_ind = 0 #int (epoch_single_subframe / crop_range)
    #        w_ind = 0 #epoch_single_subframe % crop_range
    #        size = 224 #target_size - crop_range + 1
        
    #        #shuffle data upon reset
    #        #dataGen.reset()
    #        iter_in_epoch = 0

    #        for X,Y in dataGen:
    #            #print ("len(dataGen):",str(len(dataGen)))
    #            X_subframe = X [ :, h_ind:h_ind+size, w_ind:w_ind+size, : ]
    #            #print ("X_subframe.shape,X.shape,h_ind,w_ind,size",X_subframe.shape,X.shape,h_ind,w_ind,size)
    #            model.fit ( X_subframe, Y, verbose=0 )
            
    #            iter_in_epoch += 1
    #            if iter_in_epoch >= len(dataGen):
    #                break

    #        #print ("full_epoch, epoch_single_subframe:",time.strftime("%H:%M:%S"), full_epoch, epoch_single_subframe )
    #    #e_v1.eval(model)
    #    print ("FIT START ", time.strftime("%H:%M:%S"))
    #    e_v2.eval(model, target_size=crop_size)
    #    e_v2.eval(model, target_size=crop_size, test=True)
    #    print ("FIT END ", time.strftime("%H:%M:%S"))
    ##print (time.strftime("%H:%M:%S"))

    return model