# Prepares a simple model
#   Sample downloaded from https://elitedatascience.com/keras-tutorial-deep-learning-in-python
#
# To run: 
#   model = m_det_v13.prepModel ( itarget_size=224 )

from keras.models import Model
from keras.layers import Dense, Dropout, Reshape, Flatten
from keras.optimizers import SGD
from keras.metrics import top_k_categorical_accuracy
from keras.applications.vgg16 import VGG16
import numpy as np

def prepModel( \
    D1_size = 4096, \
    D2_size = 4096, \
    cnt_classes = 200, subdiv = 3, target_size=224 ) :

    # Calculate the depth of y_hat: 205 = Pr(Obj), bx, by, bw, bh, Pr(cl_1|Obj),...Pr(cl_200|Obj)
    y_depth = cnt_classes + 5

    # Load pretrained model - except the last softmax layer
    base_model = VGG16( include_top=False, input_shape=(target_size,target_size,3) )

    # train only the top layers 
    for layer in base_model.layers:
        layer.trainable = False

    fl = Flatten()( base_model.layers[-1].output )

    # Add 2 dense and final [ subdiv * subdiv * (205) ] layer, where 205 = Pr(Obj), bx, by, bw, bh, Pr(cl_1|Obj),...Pr(cl_200|Obj)
    d1 = Dense( D1_size , activation='relu', name="D1")( fl )

    d2 = Dense( D2_size , activation='relu', name="D2")( d1 )

    d3 = Dense( subdiv**2 * y_depth , activation='sigmoid', name="D3")( d2 )

    # Reshape predictions [ subdiv; subdiv; 205 ] 
    y_pred = Reshape((subdiv, subdiv, y_depth), name='predictions')(d3)

    # this is the model we will train
    model = Model( inputs=base_model.input, outputs=y_pred )


    # First, the calc the error of Pr(Obj)

 
    optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

    #def top_5(y_true, y_pred):
    #    return top_k_categorical_accuracy(y_true, y_pred, k=5)

    model.compile( \
        #loss='mse',\
        loss=loss_det,\
        optimizer=optimizer )
        #metrics=['accuracy'])

    return model

# Define custom loss function
def loss_det(y_true, y_pred):
    print ("y_true.shape, y_pred.shape:",y_true.shape, y_pred.shape)
    assert ( len( y_true.shape ) == 4 ) # make sure it's all samples, not per sample
    m = y_true.shape[0] # number of samples

    # Pr(obj) loss
    Loss_pr_obj = np.sum ( np.square ( y_true[:,:,:,0] - y_pred[:,:,:,0] ) )

    # Only penalize bounding box and classification if there is a object in a given subdivision
    # Obj_exists is 3-dim array, indexes of items in axis of samples, subdiv-x and subdiv-y
    Obj_exists = np.where ( y_true[:,:,:,0] > .999 ) #avoid numeric instability
    
    # Pr(bbox) loss
    Loss_bbox = np.sum ( np.square ( \
        y_true[ Obj_exists[0], Obj_exists[1], Obj_exists[2], 1:5] - \
        y_pred[ Obj_exists[0], Obj_exists[1], Obj_exists[2], 1:5] ) )

    # Pr(class_i|obj) losss
    Loss_class = np.sum ( np.square ( \
        y_true[ Obj_exists[0], Obj_exists[1], Obj_exists[2], 5:] - \
        y_pred[ Obj_exists[0], Obj_exists[1], Obj_exists[2], 5:] ) )

    # Give weights: average per coordinate and average per class
    Loss = (Loss_pr_obj + Loss_bbox/4 + Loss_class/200) / m

    return Loss

#def top_5(y_true, y_pred):
#    return top_k_categorical_accuracy(y_true, y_pred, k=5)
