# Visualizes DETECTION predictions:
#   Draw bounding boxes and top-1 class name on top of images
#   Saves a single-batch files to C:\labs\KerasImagenetFruits\Visuals\DET_predictions
# 
# To run:
#   cd C:\labs\KerasImagenetFruits\KerasTrainImagenet
#   python
#   exec(open("reimport.py").read())
#   model = load_model ('d:\ilsvrc14\models\model_v62.h5', custom_objects={'loss_det': m_det_v13.loss_det})
#   exec(open("visualDetPred.py").read())

from DataGen_DET import AugSequence_v7_det_simplest as as_det_v7
from Model_DET import Model_v13_det_simplest as m_det_v13

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import pickle
from PIL import Image

# Where to save visuals with predicted bounding boxes
visuals_folder = 'C:\labs\KerasImagenetFruits\Visuals\DET_predictions'

# Where to load images from 
img_data_dir = 'C:\ILSVRC14\ILSVRC2014_DET_train_unp'

# Get DET categories, create a list from dictionary so it can be indexed
det_cats = pickle.load( open('d:\ILSVRC14\det_catdesc.obj', 'rb') )
det_cats_lst = [ det_cats[cat_name][1] for cat_name in det_cats.keys() ]

# Set target size of image
target_size = 150

# Probability of object in subdivision threshold
Pr_obj_threshold = 0.15

datasrc = "ilsvrc14_DET"

# Prepare data generator
try:
    _ = dataGen is None
except:
    dataGen = as_det_v7.AugSequence ( target_size=target_size, \
        #crop_range=crop_range, allow_hor_flip=True, 
        batch_size=64, 
        #subtractMean=subtractMean, pca_eigenvectors=pca_eigenvectors, pca_eigenvalues=pca_eigenvalues, \
        #preprocess="vgg"
        subdiv=19,
        datasrc=datasrc, test=False, debug=True )


def getRectCoord (subdiv_x, subdiv_y, bx_rel, by_rel, subdiv_width, subdiv_height, bw_rel, bh_rel, img_width, img_height):
    # Center (x, y), and Size (with, height) - absolute in whole image
    bx_abs = (subdiv_x + bx_rel) * subdiv_width
    by_abs = (subdiv_y + by_rel) * subdiv_height
                    
    #Temporary changed so that sigmoid works for image size
    bw_abs = bw_rel * img_width
    bh_abs = bh_rel * img_height
    #bw_abs = bw_rel * subdiv_width
    #bh_abs = bh_rel * subdiv_height

    # Rectangular coordinates
    x_rect = bx_abs - bw_abs/2
    y_rect = by_abs - bh_abs/2

    return (x_rect, y_rect, bw_abs, bh_abs)


# Prepare predictions
for X, y in dataGen:
    filenames = dataGen.getBatchFilenames()

    y_pred = model.predict(X)

    for img_index in range(X.shape[0]):

        # Read the actual image
        #X_single = np.asarray ( Image.open( '\\'.join ( [ img_data_dir, filenames[img_index] ] ) ) )
        X_single = X[img_index,:,:,:]
        #print (filenames[img_index], X_single.shape)

        #X_single = X[img_index,:,:,:]
        y_pred_single = y_pred[img_index,:,:,:]
        y_single = y[img_index,:,:,:]

        # to "un-vgg-preprocess_input - add mean RGB"
        #X_single = ( np.flip(X_single, axis=2) + [123.68, 116.779, 103.939]).astype(int)

        # Subdivision sizes (for now - size of X; later consider to actual file's dimensions)
        img_height, img_width = X_single.shape[0], X_single.shape[1]
        subdiv_width = img_width / y_pred_single.shape [0]
        subdiv_height = img_height / y_pred_single.shape [1]

        #plt.imshow(X_single)
        fig,ax = plt.subplots(1)

        # Remove x, y axis labels and marks
        _ = ax.set_xticklabels([])
        _ = ax.set_yticklabels([])
        _ = ax.set_xticks([])
        _ = ax.set_yticks([])

        # Display the image
        ax.imshow(X_single)

        #print ("y_pred_single.shape:", y_pred_single.shape)
        for subdiv_x in range(y_pred_single.shape[0]):
            for subdiv_y in range(y_pred_single.shape[1]):
                # Create a Rectangle patch

                # Probability of the object in the subdivision
                Pr_obj = y_pred_single [ subdiv_x, subdiv_y, 0 ]
                Pr_true_obj = y_single [ subdiv_x, subdiv_y, 0 ]

                # Highest class probability
                if y_pred_single.shape[2]>5:
                    Pr_top_class = np.argmax( y_pred_single [ subdiv_x, subdiv_y, 5: ] )
                    Pr_top_class_name = det_cats_lst[Pr_top_class]

                # Center (x, y), and Size (with, height) of bounding box - relative to subdivision
                bx_rel, by_rel, bw_rel, bh_rel = y_pred_single [ subdiv_x, subdiv_y, 1:5 ]
                bx_rel_true, by_rel_true, bw_rel_true, bh_rel_true = y_single [ subdiv_x, subdiv_y, 1:5 ]

                # Draw true bboxes in green 
                if Pr_true_obj > 0.99:
                    (x_rect_true, y_rect_true, bw_abs_true, bh_abs_true) = getRectCoord \
                        (subdiv_x, subdiv_y, bx_rel_true, by_rel_true, subdiv_width, subdiv_height, bw_rel_true, bh_rel_true, img_width, img_height)
                    rect = patches.Rectangle((x_rect_true,y_rect_true),bw_abs_true,bh_abs_true,linewidth=8.*Pr_true_obj,edgecolor='g',facecolor='none')

                    # Add the patch to the Axes
                    ax.add_patch(rect)

                # Draw rectangle if Pr_obj exceeds the threshold in red
                if Pr_obj > Pr_obj_threshold:
                    # Center (x, y), and Size (with, height) - absolute in whole image
                    bx_abs = (subdiv_x + bx_rel) * subdiv_width
                    by_abs = (subdiv_y + by_rel) * subdiv_height
                    
                    #Temporary changed so that sigmoid works for image size
                    bw_abs = bw_rel * img_width
                    bh_abs = bh_rel * img_height
                    #bw_abs = bw_rel * subdiv_width
                    #bh_abs = bh_rel * subdiv_height

                    # Rectangular coordinates
                    x_rect = bx_abs - bw_abs/2
                    y_rect = by_abs - bh_abs/2

                    rect = patches.Rectangle((x_rect,y_rect),bw_abs,bh_abs,linewidth=4.*Pr_obj,edgecolor='r',facecolor='none')

                    # Add the patch to the Axes
                    ax.add_patch(rect)

                    #Just above the rectangle write a top class name
                    if y_pred_single.shape[2]>5:
                        ax.text( x_rect,y_rect, Pr_top_class_name, fontsize=12, verticalalignment='bottom', color="red")
            
        #plt.show()

        # Save plot to a file (instead of saving to a subfolder - add subfolder to the beginning of the file

        plt.savefig ( '\\'.join ( [visuals_folder, filenames[img_index].replace('\\','__') ] ) )
        plt.close()
    # Produce visuals just for a single batch
    break