# Visualizing interim layers
#
# Functions within this file:
#   getHighestActivations - Calculates 90 random activations (features) and 
#                           returns 16 highest-activation-having-images' related patches for each feature
#   defineFigure - defines a figure for display with sub-plots for each activation and sub-sub-plots for image patches
#   updateFigure - updates the figure and saves corresponding interim-layers to file

# To run: 
#   cd C:\labs\KerasImagenetFruits\KerasTrainImagenet
#   python
#   1.have a trained model or load from file (bellow). E.g. model = load_model("D:\\ILSVRC14\\models\\model_v40.h5")
#   2.exec(open("visualNthLayer.py").read())

from keras.backend import function
import numpy as np  
import math
import matplotlib.pyplot as plt
from DataGen import DataGen_v1_150x150_1frame as dg_v1 


def getHighestActivations ( model, layer_index, map_to_image_patch_multiplier, map_to_image_patch_size, \
    cnt_activations = 90, cnt_images_per_activation = 16, debug = False ):
    # Calculates 90 random activations (features) and returns 16 highest-activation-having-images' related patches for each feature
    #
    #   model - already trained Keras model
    #   layer_index - sequence number of the desired activations layer
    #   map_to_image_patch_multiplier, map_to_image_patch_size - for input patch's calculation
    #   cnt_activations - number of random activations (features) to get max-having-images for
    #   cnt_images_per_activation - number of images per feature to return
    # Returns: tuple of
    #   high_activation_values - highest activation values np.array[90,16]
    #   high_activation_imgs - highest-activation-having-images' related patches for display list [90][16]
    #   

    # Prepare data generator
    dataGen = dg_v1.prepDataGen( target_size=224, test = False, batch_size = 64, datasrc="ilsvrc14")

    # Activation layer's dimensions (# 0th dim is #samples)
    activation_dim = (int ( model.layers [ layer_index ].output.shape[1] ), \
                      int ( model.layers [ layer_index ].output.shape[2] ), \
                      int ( model.layers [ layer_index ].output.shape[3] ) )

    if debug:
        #Print basic info about the activation:
        print ("Activation layer:", model.layers [ layer_index ].name, activation_dim )

    ## For reproducability
    #np.random.seed(111)

    # Get a function to calculate output of the layer
    func_activation = function ( [ model.input ], [ model.layers [ layer_index ].output ] )

    # 1.Randomly pick 90 features from shape of layer - will later display 3x3 activations in a single figure, 10 times
    h_activation = np.random.randint ( 0, activation_dim [ 0 ], cnt_activations )
    w_activation = np.random.randint ( 0, activation_dim [ 1 ], cnt_activations )
    c_activation = np.random.randint ( 0, activation_dim [ 2 ], cnt_activations )

    # 2.Calulate image patch locations based on features randomly selected (bellow formulas work with no padding)
    h_img_patch_start = h_activation * map_to_image_patch_multiplier
    h_img_patch_end = h_img_patch_start + map_to_image_patch_size
    w_img_patch_start = w_activation * map_to_image_patch_multiplier
    w_img_patch_end = w_img_patch_start + map_to_image_patch_size
    
    if debug:
        #Print sample feature and it's corresponding image patch:
        print ("Activation[0]: (", h_activation[0], w_activation[0], c_activation[0], \
            "); Image patch[0]: [", h_img_patch_start[0], ":", h_img_patch_end[0], "],[",\
            w_img_patch_start[0], ":" ,w_img_patch_end[0], "]" )

    # 3. Initialize structure to hold max activations values and image pathes having those activations (for display later) 
    #    list of 16 max activation values (for each of 90 pixels randomly selected)
    high_activation_values = np.zeros ( ( cnt_activations, cnt_images_per_activation ) )
    #    list of 16 images (with max activations) to be later displayed
    high_activation_imgs = [ [ np.random.rand ( map_to_image_patch_size, map_to_image_patch_size , 3 ) for i in range ( cnt_images_per_activation ) ] for i in range ( cnt_activations ) ]
    #    how many images activated the given feature?
    high_activation_cnts = np.zeros ( cnt_activations, int )

    # 4. Loop through data to get highest activations
    iter_in_epoch = 0
    for X, y in dataGen:

        # 4A. Calculate activation layer's output
        output_activation = func_activation ( [ X ] ) [0] 

        # 4B. Acumulate highest activation values and corresponding picture patches into array
        for activation_index in range ( cnt_activations ):
            for img_index in range ( X.shape [0] ):
                high_activation_value_min_index = np.argmin ( high_activation_values [ activation_index , : ] )
                activation_value = output_activation [ \
                    img_index, \
                    h_activation [ activation_index ], \
                    w_activation [ activation_index ], \
                    c_activation [ activation_index ] ]

                #Replace if current image's activation value is higher
                if activation_value > high_activation_values [ activation_index, high_activation_value_min_index ]:
                    #Print if higher activation found
                    #if debug and activation_index==0:
                    #    print ("img_index, iter_in_epoch, old_value, new_value:",\
                    #        img_index, iter_in_epoch,\
                    #        high_activation_values [ activation_index, high_activation_value_min_index ],\
                    #        activation_value)
                    # Replace activation value
                    high_activation_values [ activation_index, high_activation_value_min_index ] = activation_value
                    # Replace image patch
                    high_activation_imgs [ activation_index ] [ high_activation_value_min_index ] = np.copy ( \
                        X [ img_index,\
                        h_img_patch_start [ activation_index ] : h_img_patch_end [ activation_index ] ,\
                        w_img_patch_start [ activation_index ] : w_img_patch_end [ activation_index ] ,\
                        : ] )
                    # Increase counter of images
                    high_activation_cnts [ activation_index ] += 1
                    
        iter_in_epoch += 1
        if debug and iter_in_epoch % 50 == 0:
            print ("iter_in_epoch:",iter_in_epoch)
        if iter_in_epoch >= len(dataGen):
            break

    # Eliminate activations which don't have at least 16 images which activate them
    #print ("high_activation_cnts:",high_activation_cnts)
    activation_whereimagesexist_indexes = np.where ( high_activation_cnts >= cnt_images_per_activation )[0]
    high_activation_values = high_activation_values [ activation_whereimagesexist_indexes, : ]
    high_activation_imgs = [ np.copy ( high_activation_imgs [i] ) for i in activation_whereimagesexist_indexes ]

    return ( high_activation_values, high_activation_imgs )

def defineFigure ( activations_shape = (3,3), imgs_per_activation_shape = (4,4), patch_size = (5,5,3)  ):
    # Defines a figure for display with sub-plots for each activation and sub-sub-plots for image patches
    #
    #   activations_shape - how many activations in one figure (2-tuple)
    #   imgs_per_activation_shape - how many image patches per activation (2-tuple)
    #   patch_size - size of patch (3-tuple)
    #
    # Returns: cache - dictionary with keys:
    #   fig - pyplot figure
    #   ims - list [9][16] [np array shape of patch]

    imgs_per_activation_rows = imgs_per_activation_shape[0]
    imgs_per_activation_cols = imgs_per_activation_shape[1]
    activation_rows = activations_shape[0]
    activation_cols = activations_shape[1]
    ims = [ [ None for i in range ( imgs_per_activation_rows * imgs_per_activation_cols ) ] for i in range ( activation_rows * activation_cols ) ]
    plt.ion()
    fig = plt.figure( figsize = (\
        imgs_per_activation_cols * activation_cols /2,\
        imgs_per_activation_rows * activation_rows /2) )
    # Remove margins
    plt.subplots_adjust(left=0., bottom=0., right=1., top=1., wspace=0., hspace=0.)

    for activation_index in range ( activation_rows * activation_cols ):
        activation_start_row = int ( activation_index / activation_cols ) * imgs_per_activation_rows
        activation_start_col = int ( activation_index % activation_cols ) * imgs_per_activation_cols
        for img_index in range( imgs_per_activation_rows * imgs_per_activation_cols):
            img_row = int ( img_index / imgs_per_activation_cols ) + activation_start_row
            img_col = int ( img_index % imgs_per_activation_cols ) + activation_start_col

            # Add subplot (single feature; single image with high activation on that feature) 
            #   Pyplot subplots are indexed left-toright, top_to_bottm, starting from 1
            position = img_row * imgs_per_activation_cols * activation_cols + img_col +1
            subplot = fig.add_subplot ( \
                activation_cols * imgs_per_activation_cols, \
                activation_rows * imgs_per_activation_rows, \
                position)
            #No labels and markings on axis
            _ = subplot.set_xticklabels([])
            _ = subplot.set_yticklabels([])
            _ = subplot.set_xticks([])
            _ = subplot.set_yticks([])
            
            # add lines to separate features
            if img_index % imgs_per_activation_cols == 0 and activation_index % activation_cols > 0:
                _ = subplot.axvline(x=-0.5, linewidth=6, color='black')
            if int (img_index / imgs_per_activation_cols) == 0 and activation_index >= activation_cols:
                _ = subplot.axhline(y=-0.5, linewidth=6, color='black')

            #add image to array (iteratively will change data of this for speed rather than replacing image)
            im = subplot.imshow( np.random.rand ( patch_size[0], patch_size[1], patch_size[2] ) ) #, cmap='gray')
            ims [ activation_index ] [ img_index ] = im

    fig.canvas.flush_events()
    
    cache = {}
    cache["ims"] = ims
    cache["fig"] = fig

    return cache

def updateFigure ( file_suffix, cache, img_patches, layerLabel ):
    # Updates the figure and saves corresponding interim-layers to file
    #
    #   cache - dictionary with keys:
    #       fig - pyplot figure to update
    #       ims - list [9][16] of subplots - links to figure's suplots
    #   img_patches - [9][16] [np array shape of patch] - new image patch data
    #   layerLabel - folder name where to save images

    ims = cache["ims"]
    fig = cache["fig"]

    for activation_index in range ( len ( img_patches) ):
        for img_index in range ( len ( img_patches [ activation_index ] ) ):
            ims [ activation_index ] [ img_index ].set_data( img_patches [ activation_index ] [ img_index ] )

    fig.canvas.flush_events()

    if file_suffix is not None:
        plt.savefig("C:\\labs\\KerasImagenetFruits\\Visuals\\"+layerLabel+"\\interim_"+str(file_suffix)+".jpg")




## 1st layer - calculate activations v40 
( high_activation_values, high_activation_imgs ) = getHighestActivations ( model=model, layer_index=0, map_to_image_patch_multiplier=2, map_to_image_patch_size=7, cnt_activations = 500, cnt_images_per_activation = 16, debug = True )
cache = defineFigure ( activations_shape = (3,3), imgs_per_activation_shape = (4,4), patch_size = (7,7,3)  )
for i in range( math.ceil (high_activation_values.shape[0]/9.0) ):
    min_index = i*9
    max_index = min ( min_index + 9, high_activation_values.shape[0] )
    updateFigure ( file_suffix=i+1, cache=cache, img_patches=high_activation_imgs[min_index:max_index], layerLabel="L1" )


# 2nd layer - calculate activations v40 
( high_activation_values, high_activation_imgs ) = getHighestActivations ( model=model, layer_index=2, map_to_image_patch_multiplier=8, map_to_image_patch_size=27, cnt_activations = 500, cnt_images_per_activation = 16, debug = True )
cache = defineFigure ( activations_shape = (3,3), imgs_per_activation_shape = (4,4), patch_size = (27,27,3)  )
for i in range( math.ceil (high_activation_values.shape[0]/9.0) ):
    min_index = i*9
    max_index = min ( min_index + 9, high_activation_values.shape[0] )
    updateFigure ( file_suffix=i+1, cache=cache, img_patches=high_activation_imgs[min_index:max_index], layerLabel="L2" )

# 3rd layer - calculate activations v40 
( high_activation_values, high_activation_imgs ) = getHighestActivations ( model=model, layer_index=4, map_to_image_patch_multiplier=16, map_to_image_patch_size=75, cnt_activations = 500, cnt_images_per_activation = 16, debug = True )
cache = defineFigure ( activations_shape = (3,3), imgs_per_activation_shape = (4,4), patch_size = (75,75,3)  )
for i in range( math.ceil (high_activation_values.shape[0]/9.0) ):
    min_index = i*9
    max_index = min ( min_index + 9, high_activation_values.shape[0] )
    updateFigure ( file_suffix=i+1, cache=cache, img_patches=high_activation_imgs[min_index:max_index], layerLabel="L3" )

# 4th layer - calculate activations v40 
( high_activation_values, high_activation_imgs ) = getHighestActivations ( model=model, layer_index=5, map_to_image_patch_multiplier=16, map_to_image_patch_size=107, cnt_activations = 500, cnt_images_per_activation = 16, debug = True )
cache = defineFigure ( activations_shape = (3,3), imgs_per_activation_shape = (4,4), patch_size = (107,107,3)  )
for i in range( math.ceil (high_activation_values.shape[0]/9.0) ):
    min_index = i*9
    max_index = min ( min_index + 9, high_activation_values.shape[0] )
    updateFigure ( file_suffix=i+1, cache=cache, img_patches=high_activation_imgs[min_index:max_index], layerLabel="L4" )

# 5th layer - calculate activations v40 
( high_activation_values, high_activation_imgs ) = getHighestActivations ( model=model, layer_index=5, map_to_image_patch_multiplier=16, map_to_image_patch_size=139, cnt_activations = 500, cnt_images_per_activation = 16, debug = True )
cache = defineFigure ( activations_shape = (3,3), imgs_per_activation_shape = (4,4), patch_size = (139,139,3)  )
for i in range( math.ceil (high_activation_values.shape[0]/9.0) ):
    min_index = i*9
    max_index = min ( min_index + 9, high_activation_values.shape[0] )
    updateFigure ( file_suffix=i+1, cache=cache, img_patches=high_activation_imgs[min_index:max_index], layerLabel="L5" )
