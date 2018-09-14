# Trains a model for 50 epochs: 8x8 shifts
#
# To run:
#   model = t_v7.trainModel()

from DataGen import DataGen_v1_150x150_1frame as dg_v1 
from Model import Model_v2_addDropout as m_v2
import time
from Evaluation import Eval_v1_simple as e_v1
from Evaluation import Eval_v2_top5accuracy as e_v2

def trainModel( model = None):
    # Trains a model
    #   model = optional parameter; creates new if not passed; otherwise keeps training
    # Returns: 
    #   model: trained Keras model

    target_size=157
    dataGen = dg_v1.prepDataGen(target_size)

    if model is None:
        model = m_v2.prepModel()

    full_epochs = 1 # 1 epoch is full pass of data over all variants of 8x8 shifts
                    #  8x8x2 = 128 passes through original images in 1 full epoch
    crop_range = 8 # number of pixels to crop image (if size is 157, crops are 0-149, 1-150, ... 7-156)

    # full epoch is 8x8 = 64 passes over data: 1 times for each subframe
    for full_epoch in range (full_epochs):

        # for each subframe within a image...
        for epoch_single_subframe in range ( crop_range * crop_range ):

            #Pick a starting pixel 
            h_ind = int (epoch_single_subframe / crop_range)
            w_ind = epoch_single_subframe % crop_range
            size = target_size - crop_range + 1
        
            #shuffle data upon reset
            dataGen.reset()
            iter_in_epoch = 0

            for X,Y in dataGen:
                #print ("len(dataGen):",str(len(dataGen)))
                X_subframe = X [ :, h_ind:h_ind+size, w_ind:w_ind+size, : ]
                #print ("X_subframe.shape,X.shape,h_ind,w_ind,size",X_subframe.shape,X.shape,h_ind,w_ind,size)
                model.fit ( X_subframe, Y, verbose=0 )
            
                iter_in_epoch += 1
                if iter_in_epoch >= len(dataGen):
                    break

            print ("full_epoch, epoch_single_subframe:",time.strftime("%H:%M:%S"), full_epoch, epoch_single_subframe )
        #e_v1.eval(model)
        #e_v2.eval(model)
        #e_v2.eval(model, test=True)

    return model