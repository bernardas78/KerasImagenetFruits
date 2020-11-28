# Region proposal using Selective search

# To run:
#   cd C:\labs\KerasImagenetFruits\PreprocessImages
#   python
#   exec(open("selectiveSearch_singleFolder.py").read())

import os
from PIL import Image
#import sys
import cv2
import numpy as np
from matplotlib import colors

img_folder = 'D:\\Startup\\Visuals\\SCO_Pics\\cropped'
processed_folder = 'D:\\Startup\\Visuals\\SCO_Pics\\selectiveSearch'

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

s_color = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor () 
s_fill = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill () 
s_size = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize () 
s_texture = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture () 
s_all = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple ( s_color, s_fill, s_size, s_texture )

ss.addStrategy (s_all)

rect_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'b', 'g']

base_not_set=True

for root,dirs,files in os.walk(img_folder):

    for file in files:
        #print("file:",os.path.join(img_folder,file))

        #img = Image.open( os.path.join(img_folder,file) )
        img = cv2.imread( os.path.join(img_folder,file) )

        #if base_not_set:
        #    base_not_set = False
        ss.clearImages()
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchQuality()

        #ss.clearImages()
        #ss.addImage(img)

        #image processing
        rects = ss.process()

        # draw top N rectangles
        max_rects = 100
        for rect_id, rect in enumerate(rects):
            
            # if no more rects found
            if rect_id >= max_rects:
                break

            x, y, w, h = rect

            color_id = rect_id % 8 #up to 8 base colors available: ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
            rect_color = colors.BASE_COLORS [ list ( colors.BASE_COLORS.keys() ) [color_id] ] 
            #convert float tuple to int tuple
            rect_color = tuple( [int(255*x) for x in rect_color] )

            cv2.rectangle ( img=img, pt1=(x, y), pt2=(x+w, y+h), \
                #color=(255,0,0), \
                color=rect_color, \
                #thickness= int((max_rects-rect_id)/2)
                thickness=1
               )

        cv2.imwrite ( os.path.join(processed_folder,file), img )