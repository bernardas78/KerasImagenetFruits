# To run:
#   cd C:\labs\KerasImagenetFruits\PreprocessImages
#   python
#   exec(open("backgroundRemove_all.py").read())

import numpy as np
import cv2 as cv
import os

#path = 'D:\\Startup\\4.cut\\00001134012'  #maiselis
#path = 'D:\\Startup\\4.cut\\475002589796' #ledai
#path = 'D:\\Startup\\4.cut\\474003600555' #sviestas
path = 'D:\\Startup\\5.Filtered\\475002589796' #ledai

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        files.append(os.path.join(r, file))



#backSub = cv.createBackgroundSubtractorMOG2(history=1, varThreshold=160, detectShadows = False)
backSub = cv.createBackgroundSubtractorKNN(history=1, dist2Threshold=10000., detectShadows = False)


#capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
#if not capture.isOpened:
#    print('Unable to open: ' + args.input)
#    exit(0)
for img in files:
    #ret, frame = capture.read()

    frame = cv.imread(img)
    
    fgMask = backSub.apply(frame)


    y_mask=np.where ( fgMask>0) [0]
    x_mask=np.where ( fgMask>0) [1]
    fgMask_sparseM = np.stack ((x_mask,y_mask), axis=1)

    #draw a bounding box around white space
    #rect = cv.minAreaRect(fgMask_sparseM)
    #(x,y),(w,h), a = rect # a - angle

    rect1 = cv.boundingRect(fgMask_sparseM)
    (x_left, y_top, width, height) = rect1
    x = x_left + width/2
    y = y_top + height/2
    w = width
    h = height
    
    print (fgMask.shape, int(y-h/2), int(x-w/2), int(y+h/2), int(x+w/2))
    #cv.rectangle (img=fgMask, pt1=( int(x-w/2), int(y-h/2) ), pt2=( int(x+w/2), int(y+h/2) ), color=(255,0,0) )
    cv.rectangle (img=fgMask, pt1=( int(x-w/2), int(y-h/2) ), pt2=( int(x+w/2), int(y+h/2) ), color=(255,255,255) )

    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    frame_index = img.split('\\')[-1].split('_')[-1].split('.')[0]

    cv.putText (frame, frame_index, (15,15), cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,255))
    #cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
    #           cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0) )
    
    
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)

    _ = cv.waitKey(500)
    if frame_index=="10":
        key=input("Press Enter to continue or q to quit...")    
        if key=="q":
            break
    #if keyboard == 'q' or keyboard == 27:
    #    break