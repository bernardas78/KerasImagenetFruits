# Reads bounding boxes (Train|Test) and makes a single file
# To run:
#   cd C:\labs\KerasImagenetFruits\PreprocessImages
#   python
#   Set train or validation  
Test = True
#   exec(open("bboxesToListToFile.py").read())


import time
import random
import os
import pickle

# Pascal VOC format parser for a single folder
exec(open("voc\\voc.py").read())

if Test:
    # Validation bounding boxes
    bboxDir = "d:\\ILSVRC14\\ILSVRC2013_DET_bbox_val\\"
    picDir = "d:\\ILSVRC14\\ILSVRC2013_DET_val\\"
    detBoxesFile = "d:\\ILSVRC14\\det_bboxes_val.obj"
else:
    # Train bounding boxes
    bboxDir = "d:\\ILSVRC14\\ILSVRC2014_DET_bbox_train\\"
    picDir = "d:\\ILSVRC14\\ILSVRC2014_DET_train_unp\\"
    detBoxesFile = "d:\\ILSVRC14\\det_bboxes_train.obj"


#bboxes = []
bboxes = {}

def getSingleDirAnnotations (bboxDir, picDir, bboxes):

    # Initialize a list to accumulate bounding boxes for all files
    #bboxes = []

    # parse_voc_annotation() requires a name for cache file which must be unique (otherwise, subsequent calls don't work)
    cache_name = "cache_" + str(random.randint (0,1000000))
    ann = parse_voc_annotation (bboxDir, picDir, cache_name)

    # Combine bboxes from each file into a single list
    for singlefile in ann[0]:
        imgfilename = "\\".join( singlefile['filename'].split("\\")[ -2: ]) + ".JPEG"
        #bboxes += [ (imgfilename, obj['name'], obj['xmin'], obj['xmax'], obj['ymin'], obj['ymax'] ) for obj in singlefile['object']]
        bboxes [imgfilename] = [ ( obj['name'], obj['xmin'], obj['xmax'], obj['ymin'], obj['ymax'] ) for obj in singlefile['object']]

    # remove junk left by parse_voc_annotation() 
    os.remove(cache_name)

    #return bboxes



for _,subdirs,files in os.walk(bboxDir):
    # Train bounding boxes in sub dirs
    for subdir in subdirs:

        now=time.time()
        #bboxes += getSingleDirAnnotations ( ''.join( [ bboxDir,subdir,"\\" ] ), ''.join( [ picDir,subdir,"\\" ] ) ) 
        getSingleDirAnnotations ( ''.join( [ bboxDir,subdir,"\\" ] ), ''.join( [ picDir,subdir,"\\" ] ), bboxes ) 
        print ("parse_voc_annotation for ", subdir, " took %.2f" % (time.time()-now) )

    # Validation bounding boxes in root dir
    if len(subdirs) == 0:
        print ("PROCESSING INDIVIDUAL FILES", len(subdirs), len(files))
        now=time.time()
        #bboxes = getSingleDirAnnotations ( bboxDir, picDir )
        getSingleDirAnnotations ( bboxDir, picDir, bboxes )
        print ("parse_voc_annotation for root took %.2f" % (time.time()-now) )

    # Without this break it goes into subdirs
    break

# Save bounding boxes to a single file
with open(detBoxesFile, 'wb') as file_bboxes:
    pickle.dump(bboxes, file_bboxes)

# To load:
#   import pickle
#   bboxes = pickle.load( open("d:\\ILSVRC14\\det_bboxes_train.obj", 'rb') )
#   bboxes = pickle.load( open("d:\\ILSVRC14\\det_bboxes_val.obj", 'rb') )
