# Reads DET (Train|Test) file names and makes a single file
# To run:
#   cd C:\labs\KerasImagenetFruits\PreprocessImages
#   python
#   Set train or validation  
Test = True
#   exec(open("detFilenamesToListToFile.py").read())


import time
import random
import os
import pickle

if Test:
    # Validation pictures
    picDir = "d:\\ILSVRC14\\darknet_data\\JPEGImages\\ILSVRC2013_DET_val\\"
    picnamesFile = "d:\\ILSVRC14\\det_img_val.obj"
else:
    # Train pictures
    picDir = "d:\\ILSVRC14\\darknet_data\\JPEGImages\\ILSVRC2014_DET_train_unp\\"
    picnamesFile = "d:\\ILSVRC14\\det_img_train.obj"


def getSingleDirFilenames (picDir, subdir):

    # Initialize a list to accumulate bounding boxes for all files
    pic_filenames = []
    #print (''.join( [picDir,subdir] ))
    # Combine bboxes from each file into a single list
    for _,_,files in os.walk( ''.join( [picDir,subdir] ) ):
        for single_filename in files:
            
            pic_filenames += [''.join( [subdir, single_filename] )]

    return pic_filenames

pic_filenames = []

for _,subdirs,files in os.walk(picDir):
    # Train pics in sub dirs
    for subdir in subdirs:
        pic_filenames += getSingleDirFilenames (picDir, subdir + "\\" )
        print ("PROCESSED ", subdir)

    # Validation pics in root dir
    if len(subdirs) == 0:
        print ("PROCESSING INDIVIDUAL FILES", len(subdirs), len(files))
        now=time.time()
        pic_filenames += getSingleDirFilenames (picDir, "" )
        print ("PROCESSED ROOT" )

    # Without this break it goes into subdirs
    break

# Save bounding boxes to a single file
with open(picnamesFile, 'wb') as file_picnames:
    pickle.dump(pic_filenames, file_picnames)

# To load:
#   import pickle
#   img_train_files = pickle.load( open("d:\\ILSVRC14\\det_img_train.obj", 'rb') )
#   img_val_files = pickle.load( open("d:\\ILSVRC14\\det_img_val.obj", 'rb') )
