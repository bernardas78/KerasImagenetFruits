# Backgroud removal based item localization in bagging area
#   Used to produce Visuals/BackgroundSubtr

# To run:
#   cd C:\labs\KerasImagenetFruits-master\PreprocessImages
#   python
#   exec(open("backgroundRemove_video.py").read())

#from __future__ import print_function
import numpy as np
import cv2 as cv
#import argparse

class videoStateEnum:
    NO_MOTION=0
    IN_MOTION=1

### DEFINE THRESHOLDS ###
# how many pixels (at least) have to be classified as foreground in order to infere a motion
motion_threshold_knn = 10 #for KNN
motion_threshold_mog = 1000 #for MOG2
# how many quiet frames to encounter before exiting IN_MOTION state
quiet_frames_threshold = 3
# how many moving frames to encounter before exiting NO_MOTION state
moving_frames_threshold_knn = 1
moving_frames_threshold_mog = 2
# define area of interest (bagging area)
interest_area = (380, 590)
# should filtered or full mask be displayed?
display_filtered_mask = False

# initialize variables
##################################
video_state=videoStateEnum.NO_MOTION
quiet_frames = 0
moving_frames = 0
# should show bounding rectangle around detected item mask?
show_bounding_detected = True
#bgndImage=None
fgMask_detected_item=None
##############################





#parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
#                                              OpenCV. You can process both videos and images.')
#parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='D:\\Google Drive\\Video\\VID_20181212_171901.mp4')
#parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', 
#                    default='c:\\video\\Video_20190514201400592.avi')
#                    default='D:\\Dropbox\\AK Dropbox\\Data\\Transaction video\\Video_20190514201400592.avi')
#parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='KNN')
#args = parser.parse_args()

#input_video = 'c:\\video\\Video_20190514201400592.avi'
input_video = 'D:\\Dropbox\\AK Dropbox\\Data\\Transaction video\\Video_20190514201400592.avi'

# Uncomment one or the other
algorithm = 'KNN'
#algorithm = 'MOG2'

mog2_varThreshold = 1200 # other values tried: 800, 1000
knn_dist2Threshold = 4000 # other values tried: 400, 800, 1600, 2500

if algorithm == 'MOG2':
    agileBackSub = cv.createBackgroundSubtractorMOG2(history=1, varThreshold=160, detectShadows = False)
    conservativeBackSub = cv.createBackgroundSubtractorMOG2(history=10, varThreshold=mog2_varThreshold, detectShadows = True)
    motion_threshold = motion_threshold_mog
    moving_frames_threshold = moving_frames_threshold_mog
else:
    agileBackSub = cv.createBackgroundSubtractorKNN(history=1, dist2Threshold=10000., detectShadows = False)
    conservativeBackSub = cv.createBackgroundSubtractorKNN(history=10, dist2Threshold=4000., detectShadows = True)
    motion_threshold = motion_threshold_knn
    moving_frames_threshold = moving_frames_threshold_knn

capture = cv.VideoCapture(cv.samples.findFileOrKeep(input_video))

# Returns bounding rectangle around the masked area
def getBoundingRectangle_usingMask( param_fgMask):
    #y_axis =np.where ( param_fgMask>127) [0]    #127=shadow; 255=foreground
    #x_axis =np.where ( param_fgMask>127) [1]
    y_axis =np.where ( param_fgMask>0) [0]    #removed shadows and scaled
    x_axis =np.where ( param_fgMask>0) [1]
    fgMask_sparseM = np.stack ((y_axis,x_axis), axis=1)

    # Get bounding retangle's pt1, pt2
    rect_2 = cv.boundingRect(fgMask_sparseM)
    (y_top, x_left, h, w) = rect_2
    y_bottom = y_top+h
    x_right = x_left+w
    pt1=( x_left, y_top )
    pt2=( x_right,y_bottom )

    # Was motion detected? 
    motion_detected = x_axis.shape[0] > motion_threshold
    num_moved_points = x_axis.shape[0]

    return (pt1, pt2, motion_detected, num_moved_points)

# Apply convolutional filter to ensure at least half of the points are categorized as foreground
def applyFilter ( param_fgMask, filterSize = 5 ):
    # param_fgMask contains values 0, 127 (shadow) and 255

    #identity filter
    #my_filter = np.array( [ [ 0,0,0],[0,1,0],[0,0,0] ] )

    my_filter = np.ones ( (filterSize,filterSize) ) / (filterSize*filterSize)

    my_delta = 0.001 #avoid float rounding mistakes

    filtered_mask = cv.filter2D ( src= param_fgMask, ddepth=-1, kernel=my_filter, delta=my_delta, borderType=cv.BORDER_CONSTANT)

    filtered_mask = np.round (filtered_mask) * 1. #leave only 0's and 1s

    return filtered_mask

if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    #crop frame to area of interest only (bagging area)
    frame_interesting = np.copy ( frame [ 0:interest_area[0], 0:interest_area[1] ] )

    # get mask (moving part)
    fgMask = agileBackSub.apply(image=frame_interesting, learningRate=-1)

    # get a bounding rectangle
    (bounding_rect_pt1, bounding_rect_pt2, motion_detected, num_moved_points) = getBoundingRectangle_usingMask(fgMask)

    if ( motion_detected ):
        print ("Motion detected. #F pixels:", num_moved_points)
        #pause if motion detected
        _ = cv.waitKey(200)

    # Set proper state; learn conservative bgdSubtractor when NO_MOTION; detect item when IN_MOTION==>NO_MOTION
    if ( motion_detected ):
        #reset quiet frames
        quiet_frames = 0
        moving_frames += 1
        # Enter IN_MOTION state
        if ( moving_frames >= moving_frames_threshold and video_state == videoStateEnum.NO_MOTION):
            video_state = videoStateEnum.IN_MOTION
            print ("NO_MOTION ==> IN_MOTION")

    if ( not motion_detected ):
        # increase quiet frames
        moving_frames = 0
        quiet_frames += 1
        # Exit IN_MOTION state if enough quiet frames detected; detect item
        if ( quiet_frames >= quiet_frames_threshold and video_state == videoStateEnum.IN_MOTION):
            video_state = videoStateEnum.NO_MOTION

            # when exiting IN_MOTION state - detect item HERE
            fgMask_detected_item = conservativeBackSub.apply(frame_interesting, learningRate=0)

            # remove shaddow (values 127) and scale to 0..1
            fgMask_detected_item = np.round (fgMask_detected_item / 255.) * 1.

            # remove small pathes of white
            fgMask_detected_filtered = applyFilter (fgMask_detected_item)

            # get bounding rectangle
            (detected_pt1, detected_pt2, _, num_moved_points) = getBoundingRectangle_usingMask(fgMask_detected_filtered)
            print ("detect item HERE. ",detected_pt1, detected_pt2, num_moved_points)
            
            # Draw a DETECTED ITEM localization on the image
            cv.rectangle(frame, detected_pt1, detected_pt2, (0,255,0) )
            cv.imshow('Frame', frame)

            # Draw detected mask (either filtered for small patches or not) and bounding rectangle 
            if (show_bounding_detected):
                if (display_filtered_mask):
                    cv.rectangle(fgMask_detected_filtered, detected_pt1, detected_pt2, (255,255,255) )
                    cv.imshow('FG Mask Filtered', fgMask_detected_filtered)
                else:
                    cv.rectangle(fgMask_detected_item, detected_pt1, detected_pt2, (255,255,255) )
                    cv.imshow('FG Mask Detected', fgMask_detected_item)
            _ = cv.waitKey(2000)

        if ( video_state == videoStateEnum.NO_MOTION ):
            # learn conservative bgndSubtractor
            conservativeBackSub.apply(frame_interesting, learningRate=-1)
            print ("learn conservative bgndSubtractor")

    # show frame number and picture itself
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))  
    cv.imshow('Frame', frame)
      
    #keyboard = cv.waitKey(30)
    #if keyboard == 'q' or keyboard == 27:
    #    break