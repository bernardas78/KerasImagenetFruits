# To run:
#   cd C:\labs\KerasImagenetFruits\PreprocessImages
#   python
#   exec(open("backgroundRemove_2imgs.py").read())

#from __future__ import print_function
import numpy as np
import cv2 as cv
#import argparse

img1='D:\\Startup\\TrainAndVal\\Train\\00001134012\\00001134012_20190513122511_1.jpg'
img2='D:\\Startup\\TrainAndVal\\Train\\00001134012\\00001134012_20190513122511_2.jpg'
img3='D:\\Startup\\TrainAndVal\\Train\\00001134012\\00001134012_20190513122512_3.jpg'
img4='D:\\Startup\\TrainAndVal\\Train\\00001134012\\00001134012_20190513122512_4.jpg'
img5='D:\\Startup\\TrainAndVal\\Train\\00001134012\\00001134012_20190513122512_5.jpg'
img6='D:\\Startup\\TrainAndVal\\Train\\00001134012\\00001134012_20190513122512_6.jpg'
img7='D:\\Startup\\TrainAndVal\\Train\\00001134012\\00001134012_20190513122513_7.jpg'
img8='D:\\Startup\\TrainAndVal\\Train\\00001134012\\00001134012_20190513122513_8.jpg'
img9='D:\\Startup\\TrainAndVal\\Train\\00001134012\\00001134012_20190513122513_9.jpg'
img10='D:\\Startup\\TrainAndVal\\Train\\00001134012\\00001134012_20190513122513_10.jpg'

imgs = [ img1, img2, img3, img4, img5, img6, img7, img8, img9, img10 ]

#backSub = cv.createBackgroundSubtractorMOG2(history=1, varThreshold=160, detectShadows = False)
backSub = cv.createBackgroundSubtractorKNN(history=1, dist2Threshold=10000., detectShadows = False)


#capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
#if not capture.isOpened:
#    print('Unable to open: ' + args.input)
#    exit(0)
for img in imgs:
    #ret, frame = capture.read()
    #if frame is None:
    #    break
    frame = cv.imread(img)
    
    fgMask = backSub.apply(frame)

    x=np.where ( fgMask>0) [0]
    y=np.where ( fgMask>0) [1]
    fgMask_sparseM = np.stack ((x,y), axis=1)

    #draw a bounding box around white space
    #rect = cv.minAreaRect(fgMask_sparseM)
    #(x,y),(w,h), a = rect # a - angle

    rect1 = cv.boundingRect(fgMask_sparseM)
    (x_left, y_top, x_right, y_bottom) = rect1
    x = ( x_right + x_left ) /2
    y = ( y_top + y_bottom ) /2
    w = ( x_right - x_left )
    h = ( y_bottom - y_top )
    
    print (fgMask.shape, int(y-h/2), int(x-w/2), int(y+h/2), int(x+w/2))
    #cv.rectangle (img=fgMask, pt1=( int(x-w/2), int(y-h/2) ), pt2=( int(x+w/2), int(y+h/2) ), color=(255,0,0) )
    cv.rectangle (img=fgMask, pt1=( int(y-h/2), int(x-w/2) ), pt2=( int(y+h/2), int(x+w/2) ), color=(255,255,255) )

    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    #cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
    #           cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    
    keyboard = cv.waitKey(500)
    #if keyboard == 'q' or keyboard == 27:
    #    break