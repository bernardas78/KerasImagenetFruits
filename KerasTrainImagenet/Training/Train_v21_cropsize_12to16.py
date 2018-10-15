# Trains a model for 4 epochs: 16x16 shifts. Size of image trainable 224x224. with shifts of 16x16, target size is 239
#
# To run:
#   model = t_v21.trainModel()

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

    crop_range = 16 # number of pixels to crop image (if size is 239, crops are 0-223, 1-224, ... 15-238)
    crop_size = 224
    target_size = crop_size + crop_range - 1 #239

    dataGen = dg_v1.prepDataGen(target_size = target_size, batch_size = 128 )

    if model is None:
        input_shape = (224, 224, 3)
        model = m_v2.prepModel ( input_shape = input_shape )

    full_epochs = 4 # 1 epoch is full pass of data over all variants of 16x16 shifts
                    #  16x16 = 256 passes through original images in 1 full epoch

    # full epoch is 16x16 = 256 passes over data: 1 times for each subframe
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
        e_v2.eval(model, target_size=crop_size)
        e_v2.eval(model, target_size=crop_size, test=True)

    return model