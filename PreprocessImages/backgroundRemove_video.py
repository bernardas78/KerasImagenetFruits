# To run:
#   cd C:\labs\KerasImagenetFruits\PreprocessImages
#   python
#   exec(open("backgroundRemove_video.py").read())

from __future__ import print_function
import numpy as np
import cv2 as cv
import argparse
parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
#parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='D:\\Google Drive\\Video\\VID_20181212_171901.mp4')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='D:\\Dropbox\\AK Dropbox\\Data\\Transaction video\\Video_20190514201400592.avi')

parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='KNN')
args = parser.parse_args()
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2(history=1, varThreshold=160, detectShadows = False)
else:
    backSub = cv.createBackgroundSubtractorKNN(history=1, dist2Threshold=10000., detectShadows = False)
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
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
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break