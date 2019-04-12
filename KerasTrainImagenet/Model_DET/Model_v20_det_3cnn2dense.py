# Prepares a simple model
#   Sample downloaded from https://elitedatascience.com/keras-tutorial-deep-learning-in-python
#
# To run: 
#   model = m_det_v20.prepModel (subdiv = 19 ) 

#from keras.activations import relu
from keras.models import Model, Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Reshape, Flatten, Concatenate, BatchNormalization, Activation
from keras.initializers import RandomNormal
from keras.optimizers import SGD
from keras.metrics import top_k_categorical_accuracy
from keras.applications.vgg16 import VGG16
import numpy as np
from keras import backend as K

def prepModel( input_shape=(150,150,3), \
    #L1_size_stride_filters = (3, 1, 32), L1MaxPool_size_stride = None,    L1_dropout = 0.0, \
    #L2_size_stride_filters = (3, 1, 64), L2MaxPool_size_stride = (2, 2),  L2_dropout = 0.0, \
    #L3_size_stride_filters = (3, 1, 128),                                 L3_dropout = 0.0, \
    #L4_size_stride_filters = (3, 1, 256), L4MaxPool_size_stride = (3, 3), L4_dropout = 0.25, \
    #L5_size_stride_filters = (3, 1, 384), L5MaxPool_size_stride = (3, 3), L5_dropout = 0.0, \
    D1_size = 4096,                                                       D1_dropout = 0.5, \
    D2_size = 2048,                                                       D2_dropout = 0.5, \
    #Conv_padding = "valid", \
    cnt_classes = 200, subdiv = 3 ) :

    # Calculate the depth of y_hat: 205 = Pr(Obj), bx, by, bw, bh, Pr(cl_1|Obj),...Pr(cl_200|Obj)
    y_depth = cnt_classes + 5

    #region Model difinition
    #L1_size = L1_size_stride_filters[0]
    #L1_stride = L1_size_stride_filters[1]
    #L1_filters = L1_size_stride_filters[2]

    #L2_size = L2_size_stride_filters[0]
    #L2_stride = L2_size_stride_filters[1]
    #L2_filters = L2_size_stride_filters[2]

    #L3_size = L3_size_stride_filters[0]
    #L3_stride = L3_size_stride_filters[1]
    #L3_filters = L3_size_stride_filters[2]

    base_model = Sequential()
 
    base_model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=input_shape))
    base_model.add(Convolution2D(32, (3, 3), activation='relu'))
    base_model.add(MaxPooling2D(pool_size=(3,3)))
    #model.add(Dropout(0.25))

    base_model.add(Convolution2D(32, (3, 3), activation='relu'))
    base_model.add(MaxPooling2D(pool_size=(3,3)))

    base_model.add(Flatten())
    #base_model.add(BatchNormalization(center=False, scale=False))

    
    #base_model.add(Dense(D1_size, activation='relu'))
    base_model.add(Dense(D1_size))
    base_model.add(Activation('relu'))

    if D1_dropout > 0.:
        base_model.add ( Dropout ( D1_dropout ) )
    
    base_model.add (Dense(D2_size, activation='relu'))
    #base_model.add(BatchNormalization())
    if D2_dropout > 0.:
        base_model.add ( Dropout ( D2_dropout ) )
    #endregion
    # Add 2 dense and final [ subdiv * subdiv * (205) ] layer, where 205 = Pr(Obj), bx, by, bw, bh, Pr(cl_1|Obj),...Pr(cl_200|Obj)

    #C_LAST_DEBUGfl = Flatten()( base_model.layers[-1].output )  

    # Add 2 dense and final [ subdiv * subdiv * (205) ] layer, where 205 = Pr(Obj), bx, by, bw, bh, Pr(cl_1|Obj),...Pr(cl_200|Obj)
    #init_d1 = RandomNormal(mean=0.0, stddev=np.sqrt(2/25088), seed=None)
    #C_LAST_DEBUGd1 = Dense( D1_size , activation='tanh', name="D1")( fl )
    #d1_dropout = Dropout(0.5)(d1)

    #d2 = Dense( D2_size , activation='relu', name="D2")( d1_dropout )
    #init_d2 = RandomNormal(mean=0.0, stddev=np.sqrt(2/D1_size), seed=None)
    #C_LAST_DEBUGd2 = Dense( D2_size , activation='tanh', name="D2")( d1 )
    #d2_dropout = Dropout(0.5)(d2)

    # Add last layer elements: Pr(obj), bbox (center+size), Pr(class)
    #init_d3_probj = RandomNormal(mean=0.0, stddev=np.sqrt(2/D2_size), seed=None)
    d3_probj = Dense( subdiv**2 * 1 , activation='sigmoid', name="d3_probj")( base_model.output )
    #C_LAST_DEBUGd3_bbox = Dense( subdiv**2 * 4 , name="d3_bbox")( d2 )
    d3_bbox = Dense( subdiv**2 * 4 , name="d3_bbox")( base_model.output )
    #d3_class = Dense( subdiv**2 * cnt_classes , activation='sigmoid', name="d3_class")( d2 ) # need be separate softmax for each subdivision

    d3_probj_resh = Reshape((subdiv, subdiv, 1), name='d3_probj_resh')( d3_probj )
    d3_bbox_resh = Reshape((subdiv, subdiv, 4), name='d3_bbox_resh')( d3_bbox )
    #d3_class_resh = Reshape((subdiv, subdiv, cnt_classes), name='d3_class_resh')( d3_class )

    #d3 = Concatenate (axis=-1)( [d3_probj, d3_bbox, d3_class] )
    d3_resh = Concatenate (axis=-1)( [d3_probj_resh, d3_bbox_resh] )
    #y_pred = Concatenate (axis=-1, name='predictions')( [d3_probj_resh, d3_bbox_resh, d3_class_resh] )
    #y_pred = d3_bbox_resh  #DET_DEBUG
    #C_LAST_DEBUGy_pred = d3_resh
    #d3_resh = Dense( subdiv**2 * 5 , activation='sigmoid', name="d3_probj")( base_model.output ) #DET_DEBUG_1
    y_pred = d3_resh

    # this is the model we will train
    model = Model( inputs=base_model.input, outputs=y_pred )

    # Default values for optimizer
    optimizer = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)

    #def top_5(y_true, y_pred):
    #    return top_k_categorical_accuracy(y_true, y_pred, k=5)

    model.compile( \
        #loss='mse',\
        loss=loss_det,\
        optimizer=optimizer)
        #optimizer='sgd')
        #metrics=['accuracy'])

    return model

# Define custom loss function
def loss_det(y_true, y_pred, print_loss=False):
    # Coefficients for loss influence:
    lambda_coord = 0.25
    lambda_noobj = 0.05

    #print ("y_true.shape, y_pred.shape:",y_true.shape, y_pred.shape)
    assert ( len( y_true.shape ) == 4 ) # make sure it's all samples, not per sample

    # Number of samples
    m = K.tf.shape (y_true)[0] 

    # Number of subdivisions
    subdiv_x = K.tf.shape (y_true)[1] 
    subdiv_y = K.tf.shape (y_true)[2] 

    # Only penalize bounding box and classification if there is a object in a given subdivision
    # Obj_exists is 3-dim array, indexes of items in axis of samples, subdiv-x and subdiv-y
    Obj_exists = K.tf.where( K.tf.greater ( y_true[:,:,:,0], .999 ) )      #DET_DEBUG
    Obj_notexists = K.tf.where( K.tf.less ( y_true[:,:,:,0], .001 ) )      #DET_DEBUG


    #print ("Shape ob Obj_exists:", K.tf.shape ( Obj_exists))
    # Count of object in all samples, all subdivisions
    Obj_count = K.tf.shape ( Obj_exists )[0]                               #DET_DEBUG

    # Create tensors of filtered y_true and y_pred tensors
    y_true_obj = K.tf.gather_nd ( y_true, Obj_exists)                      #DET_DEBUG
    y_pred_obj = K.tf.gather_nd ( y_pred, Obj_exists)                      #DET_DEBUG
    y_pred_noobj = K.tf.gather_nd ( y_pred, Obj_notexists)                 #DET_DEBUG
    
    # Separate Pr(obj) and Pr(Noobj)
    # Pr(obj) loss
    #Loss_pr_obj = K.mean ( K.square ( y_true[:,:,:,0] - y_pred[:,:,:,0] ) )  #DET_DEBUG
    Loss_pr_obj = K.sum ( K.square ( K.tf.subtract ( 1., K.tf.to_float (y_pred_obj [ :, 0 ]) ) ) )
    Loss_pr_noobj = K.sum ( K.square ( K.tf.to_float (y_pred_noobj [ :, 0 ]) ) )

    # Pr(bbox) loss - only penalize if object us there
    # Reduce significance of bbox error by taking the sqrt
    Loss_bbox = K.sqrt ( K.sum ( K.square ( K.tf.subtract ( K.tf.to_float (y_true_obj [ :, 1:5 ]), K.tf.to_float (y_pred_obj [ :, 1:5 ]) ) ) ) ) \
        / K.tf.to_float (Obj_count) / 4.
        #/ K.tf.to_float (subdiv_x * subdiv_y * m)
                       
    # Pr(class_i|obj) losss
    #Loss_class = K.sum ( K.square ( \
    #    y_true_obj [ :, 5: ] - y_pred_obj [ :, 5: ] \
        #y_true[ Obj_exists[0], Obj_exists[1], Obj_exists[2], 5:] - \
        #y_pred[ Obj_exists[0], Obj_exists[1], Obj_exists[2], 5:] ) \
    #    ) )

    # Give weights: average per coordinate and average per class
    #Loss = Loss_bbox /4. #/ K.tf.to_float (Obj_count)
    #Loss = K.mean (K.square (y_pred - y_true))
    Loss = K.tf.to_float(Loss_pr_obj) + lambda_noobj * K.tf.to_float(Loss_pr_noobj) + lambda_coord * K.tf.to_float(Loss_bbox)
    #Loss = Loss_class /200. / K.tf.to_float (Obj_count)
    
    # Pr_obj loss only
    #Loss = Loss_pr_obj / K.tf.to_float(m*subdiv_x*subdiv_y)
    
    # BBox loss only
    #Loss = (Loss_bbox/4.) / K.tf.to_float (Obj_count)

    # Class loss only
    #Loss = (Loss_class/200.) / K.tf.to_float (Obj_count)

    #print ("LOSS:", Loss)
    if print_loss:
        sess = K.get_session()
        print ("Loss_pr_obj:",sess.run(Loss_pr_obj))
        print ("Loss_pr_noobj:",sess.run(Loss_pr_noobj))
        print ("Loss_bbox:",sess.run(Loss_bbox))
        print ("Loss:",sess.run(Loss))

    return Loss

# m_det_v17.loss_det_notensor(y_true, y_pred)
def loss_det_notensor(y_true, y_pred):

    #loss = loss_det ( y_true, y_pred )
    #sess = K.get_session()
    #(evaluated_loss) = sess.run ( [loss], feed_dict = {model.input:X} )

    # Number of samples
    m = y_true.shape[0] 

    # Number of subdivisions
    subdiv_x = y_true.shape[1] 
    subdiv_y = y_true.shape[2] 

    # Pr(obj) loss
    Loss_pr_obj = np.mean ( np.square ( y_true[:,:,:,0] - y_pred[:,:,:,0] ) )
    print ("Loss_pr_obj:",Loss_pr_obj)
    #Loss_pr_obj = K.sum ( K.square ( y_true[:,:,:,:] - y_pred[:,:,:,:] ) )

    # Only penalize bounding box and classification if there is a object in a given subdivision
    # Obj_exists is 3-dim array, indexes of items in axis of samples, subdiv-x and subdiv-y
    Obj_exists = np.where( np.greater ( y_true[:,:,:,0], .999 ) )      #DET_DEBUG

    #print ("Shape ob Obj_exists:", K.tf.shape ( Obj_exists))
    # Count of object in all samples, all subdivisions
    Obj_count = len ( Obj_exists [0])                               #DET_DEBUG

    # Create tensors of filtered y_true and y_pred tensors
    y_true_obj = y_true [ Obj_exists ]                      #DET_DEBUG
    y_pred_obj = y_pred [ Obj_exists ]                      #DET_DEBUG
    
    # Pr(bbox) loss - only penalize if object us there
    # Reduce significance of bbox error by taking the sqrt
    Loss_bbox = np.sqrt ( np.sum ( np.square ( np.subtract ( y_true_obj [ :, 1:5 ], y_pred_obj [ :, 1:5 ] ) ) ) ) / Obj_count / 4.
        #/ K.tf.to_float (subdiv_x * subdiv_y * m)
    print ("Loss_bbox:",Loss_bbox)
                       
    # Pr(class_i|obj) losss
    #Loss_class = K.sum ( K.square ( \
    #    y_true_obj [ :, 5: ] - y_pred_obj [ :, 5: ] \
        #y_true[ Obj_exists[0], Obj_exists[1], Obj_exists[2], 5:] - \
        #y_pred[ Obj_exists[0], Obj_exists[1], Obj_exists[2], 5:] ) \
    #    ) )

    # Give weights: average per coordinate and average per class
    #Loss = Loss_bbox /4. #/ K.tf.to_float (Obj_count)
    #Loss = K.mean (K.square (y_pred - y_true))
    Loss = Loss_pr_obj + Loss_bbox
    return Loss