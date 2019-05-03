# Visualize a single picture and all it's activations of the interim CNN layers
#

# To run: 
#   cd C:\labs\KerasImagenetFruits\KerasTrainImagenet
#   python
#   0.exec(open("reimport.py").read())
#   1.have a trained model or load from file (bellow). E.g. model = load_model("D:\ILSVRC14\models\model_v59.h5", custom_objects={'top_5': m_v11.top_5})
#   2.exec(open("visualInterimLayersSinglePic.py").read())

from keras.backend import function
import numpy as np  
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
#from DataGen import AugSequence_v5_vggPreprocess as as_v5
#from Model import Model_v11_pretVggPlusSoftmax as m_v11
#from keras.models import load_model

def get_activations ( model, X ):
    print ( "Getting activations of X shaped ", X.shape)

    # Initialize a return value: list of activations
    list_activations = []
    list_layernames = []

    # Don't include input layers because full original picture will be included
    ## include input layer as well 
    #list_activations.append ( X )
    #list_layernames.append ( 'Input' )

    for layer in model.layers:

        # only show cnn and maxpooling layers
        if not 'conv' in layer.name and not 'pool'  in layer.name:
            continue
        print ( "Getting activations of ", layer.name )
        
        # Get a function to calculate output of the layer
        func_activation = function ( [ model.input ], [ layer.output ] )

        # Calculate activation layer's output
        output_activation = func_activation ( [ X ] ) [0]

        # Append output to a list of activations for return
        list_activations.append (output_activation)
        list_layernames.append ( layer.name )

    return (list_layernames, list_activations)

crop_range = 32 # number of pixels to crop image (if size is 235, crops are 0-223, 1-224, ... 11-234)
target_size = 224
datasrc = "ilsvrc14_100boundingBoxes"

## Prepare data generator
dataGen = as_v5.AugSequence ( target_size=target_size, crop_range=crop_range, allow_hor_flip=True, batch_size=32, \
        #subtractMean=subtractMean, pca_eigenvectors=pca_eigenvectors, pca_eigenvalues=pca_eigenvalues, \
        preprocess="vgg", datasrc=datasrc, test=False )

# Get the first batch of images
for X, y in dataGen:
    # only produce visuals for the first batch of images
    break

# Get activations of all layers, all images of a batch
(layernames_list, activation_list) = get_activations ( model=model, X=X )

grid_size = (8,8)

#image_id = 0
cnt_images = activation_list[0].shape[0]
# For each image
for image_id in range(cnt_images):

    # Create a folder for single image's interim layers pics 
    image_folder = "C:\\labs\\KerasImagenetFruits\\Visuals\\ActivationsSingleImage\\" + str(image_id)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    # Place the original picture
    # vgg-upprocess the image
    X_single = ( np.flip(X [image_id,:,:,:] , axis=2) + [123.68, 116.779, 103.939]).astype(np.uint8)
    plt.imsave (image_folder + "\\original.jpg", X_single )
    #plt.savefig(image_folder + "\\original.jpg")
    plt.close()

    # For each activation (i.e. each network's deep layer)
    for activation_index in range( len(activation_list) ):


        # For each filter within an activation, produce a 1+ image (up to 64 subplots per image)
        cnt_filters = activation_list [activation_index].shape[3]
        cnt_subplots_per_image = grid_size[0]*grid_size[1]
        total_images_for_activation = math.ceil ( cnt_filters / cnt_subplots_per_image )
        print ("activation_index: ", str(activation_index), "; cnt_filters: ", str(cnt_filters), " ; total_images_for_activation: ", str(total_images_for_activation))

        # Priduce 1+ images for each activation
        for activation_image_id in range(total_images_for_activation):

            start_filter_id = activation_image_id * cnt_subplots_per_image
            end_filter_id = min ( (activation_image_id+1) * cnt_subplots_per_image, cnt_filters)

            print ("start_filter_id, end_filter_id", start_filter_id, end_filter_id)

            # Produce a single image
            f, axarr = plt.subplots ( grid_size[0], grid_size[1], gridspec_kw={'wspace':0.0, 'hspace':0.05 } ) 
            f.set_size_inches (6,6)

            # Up to 64 subplots
            for filter_index in np.arange( start_filter_id, end_filter_id ): # shape[3] = #layers of activation

                row = math.floor ( (filter_index-start_filter_id) / grid_size[1] )
                col = (filter_index-start_filter_id) % grid_size[1]
                #print ("row, col: ", str(row), ",", str(col) )
                # Show actual activations
                axarr[row,col].imshow ( activation_list [activation_index] [image_id,:,:,filter_index] )

                # no labels or markings on axes
                _ = axarr[row,col].set_xticklabels([])
                _ = axarr[row,col].set_yticklabels([])
                _ = axarr[row,col].set_xticks([])
                _ = axarr[row,col].set_yticks([])

            print ("image: ", str(image_id), "activation:", str(activation_index), "filters:", str(start_filter_id), "-", str(end_filter_id) )
            #plt.show()
            plt.savefig(image_folder + "\\" + str( activation_index ) + "." + layernames_list[activation_index] + "_" + str(start_filter_id) + "-" + str(end_filter_id) + ".jpg")
            plt.close()