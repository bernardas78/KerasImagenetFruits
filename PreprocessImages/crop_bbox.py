# Crops images using bounding boxes provided by ILSVRC (source: http://www.image-net.org/download-bboxes; instr to unpack: D:\ILSVRC14\readme.txt)
#	Bounding box definitions: D:\ILSVRC14\Annotation_unp\Annotation
#	Source image folder: D:\ILSVRC14\ILSVRC2012_img_train_unp
#	Destination (cropped) folder: D:\ILSVRC14\ILSVRC2012_bbox
#
# To run:
#   cd C:\labs\KerasImagenetFruits\PreprocessImages
#   python
#	exec(open("crop_bbox.py").read())

from oct2py import octave
from PIL import Image
import numpy as np
import os
import sys

# using VOC file parsing from ILSVRC devkit (VOCreadxml.m)
octave.addpath('D:\ILSVRC14\ILSVRC2014_devkit\evaluation')
octave.warning('off','all')

# Bounding box definitions 
bbox_def_path = "D:\\ILSVRC14\\Annotation_unp\\Annotation\\"

# Training images are located
full_images_folder = "D:\\ILSVRC14\\ILSVRC2012_img_train_unp\\"

# Destination for cropped images
cpopped_folder = "D:\\ILSVRC14\\ILSVRC2012_bbox\\"

#print ("Looping through file names")
#for _,dirs,files in os.walk(picDir):

for _,class_folders,_ in os.walk (bbox_def_path) :
    #print("files.shape:",len(files))
    for class_folder in class_folders:

        # ILSVRC dataset consists of 1000 classes, but annotations are for 3K+ classes. Skip annotations without images
        if not os.path.exists ( full_images_folder + class_folder):
            print ("Class does not exist:", class_folder)
            continue

        # Create folder for cropped images
        if not os.path.exists ( cpopped_folder + class_folder):
            os.mkdir ( cpopped_folder + class_folder )
        for _,_,annotation_files in os.walk ( bbox_def_path + class_folder ):
            for annotation_file in annotation_files:
        
                # Read file properties
                rec = octave.VOCreadxml(bbox_def_path + "\\" + class_folder + "\\" + annotation_file)

                folder, filename = \
                    rec.annotation.folder,\
                    rec.annotation.filename

                width, height = \
                    int(rec.annotation.size.width), \
                    int(rec.annotation.size.height)

                # Due to multiple bounding boxes - loop bellow for each (referncing in octave differs)
                bbox_cnt = int(octave.size(rec.annotation.object)[0,1])
                for bbox_id in range ( bbox_cnt ):
                    #print ("bbox_id:", str(bbox_id))

                    if bbox_cnt>1:
                        xmin, ymin, xmax, ymax = \
	                        int(rec.annotation.object.bndbox[0][bbox_id].xmin), \
	                        int(rec.annotation.object.bndbox[0][bbox_id].ymin), \
	                        int(rec.annotation.object.bndbox[0][bbox_id].xmax), \
	                        int(rec.annotation.object.bndbox[0][bbox_id].ymax)
                    else:
                        xmin, ymin, xmax, ymax = \
	                        int(rec.annotation.object.bndbox.xmin), \
	                        int(rec.annotation.object.bndbox.ymin), \
	                        int(rec.annotation.object.bndbox.xmax), \
	                        int(rec.annotation.object.bndbox.ymax)


                    ##########
                    # Make a crop: try full bbox and increase smaller dimension to make square
                    ##########

                    # try to make a square of [prefered_crop_size,prefered_crop_size]
                    prefered_crop_size = np.max ( [ xmax-xmin,  ymax-ymin ]) 

                    if (xmax-xmin) > (ymax-ymin):
                        #for wider images: 
                        ##  x-bounds as specified
                        crop_xmin = xmin
                        crop_xmax = xmax
                        #   y preferred center as specified
                        y_center_prefered = ymin + (ymax-ymin)/2
                        #       try to take 1/2 prefered size from prefered center, up to 0th pixel
                        crop_ymin = np.max ( [ int(y_center_prefered - prefered_crop_size/2),  0 ] )
                        #       then try to take prefered size from crop_ymin, up to full height
                        crop_ymax = np.min ( [ crop_ymin + prefered_crop_size,  height ] )
                        #       in case bbox was in the bottom of image - push back the crop_ymin to make as square as possible, up to 0th pixel
                        crop_ymin = np.max ( [ crop_ymax - prefered_crop_size, 0 ] )
                    else:
                        #for higher images: 
                        ##  y-bounds as specified
                        crop_ymin = ymin
                        crop_ymax = ymax
                        #   x preferred center as specified
                        x_center_prefered = xmin + (xmax-xmin)/2
                        #       try to take 1/2 prefered size from prefered center, up to 0th pixel
                        crop_xmin = np.max ( [ int(x_center_prefered - prefered_crop_size/2),  0 ] )
                        #       then try to take prefered size from crop_xmin, up to full width
                        crop_xmax = np.min ( [ crop_xmin + prefered_crop_size,  width ] )
                        #       in case bbox was in the right of image - push back the crop_xmin to make as square as possible, up to 0th pixel
                        crop_xmin = np.max ( [ crop_xmax - prefered_crop_size, 0 ] )

                    #print ("xmin %1d, ymin %1d, xmax %1d, ymax %1d" % ( crop_xmin, crop_ymin, crop_xmax, crop_ymax) )

                    full_image_path = full_images_folder + folder + "\\" + filename + ".jpeg"
            
                    #Some files don't exist - skip
                    if not os.path.exists ( full_image_path):
                        print ("Image File does not exist: ", full_image_path )
                        continue

                    cpopped_path = cpopped_folder + folder + "\\" + filename + "_crop." + str(bbox_id) + ".jpeg"

                    try:
                        im = Image.open( full_image_path )

                        im_cropped = im.crop ( ( crop_xmin, crop_ymin, crop_xmax, crop_ymax ) )

                        im_cropped.save( cpopped_path )
                    except (OSError,SystemError) as e:
                        print ("Error: ", e, "file: ", filename)
                        pass