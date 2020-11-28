# To run:
#   cd C:\labs\KerasImagenetFruits\PreprocessImages
#   python
#   exec(open("sift.py").read())
#
#   To use SIFT (Scal Invariant Feature Transform - build OpenCV locally:
#       https://drthitirat.wordpress.com/2019/01/20/opencv-python-build-opencv-4-0-1-dev-contrib-non-free-siftsurf-from-sources-on-windows-10-64-bit-os/
#Theory on SIFT:
#       Lecture: https://www.youtube.com/watch?v=NPcMS49V5hg
#       Paper:


import numpy as np
import cv2 as cv

# Pastume grietine
img_path = 'D:/Startup/Visuals/SCO_Pics/cropped/vlcsnap-2019-07-25-10h43m56s230_result.png'
img2_path = 'D:/Startup/Visuals/SCO_Pics/cropped/vlcsnap-2019-07-25-10h43m59s495_result.png'
# Pastume arbata
#img_path = 'D:/Startup/Visuals/SCO_Pics/cropped/vlcsnap-2019-07-25-10h43m51s874_result.png'
#img2_path = 'D:/Startup/Visuals/SCO_Pics/cropped/vlcsnap-2019-07-25-10h43m56s230_result.png'

img = cv.imread( img_path )
#gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create()
#kp = sift.detect(gray,None)
kp = sift.detect(img,None)
#img_kp=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_kp=cv.drawKeypoints(img,kp,img_kp,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite('D:/Startup/Visuals/SCO_Pics/SIFT/sift_keypoints.jpg',img_kp)

# descriptors (tuple [KeyPoints,Descriptors]) Descriptor shape [128,] - seems 8x8 shape of [orientation, size]
_,desc=sift.compute(img,kp)

img2 = cv.imread( img2_path )
#gray2= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
#kp2, desc2 = sift.detectAndCompute(gray2,None)
kp2, desc2 = sift.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_L2 , crossCheck=False)
# Match descriptors.
matches = bf.match(desc,desc2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
#img_matches = cv.drawMatches(gray,kp,gray2,kp2,matches[:50],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_matches = cv.drawMatches(img,kp,img2,kp2,matches[:50],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#plt.imshow(img_matches)
#plt.show()
cv.imwrite('D:/Startup/Visuals/SCO_Pics/SIFT/sift_kp_matches.jpg',img_matches)

# Find matches which changed positions in image
#for match in matches:
#    # image keypoint indexes
#    desc_kp_id = match.trainIdx
#    desc2_kp_id = match.queryIdx

#    # Difference of positions in matching keypoints (>0 if item moved)
#    dist = cv.norm ( kp2[desc_kp_id].pt, kp[desc2_kp_id].pt, cv.NORM_L2 )

dist_lst = [ cv.norm ( kp[match.queryIdx].pt, kp2[match.trainIdx].pt, cv.NORM_L2 ) for match in matches ]
dist_arr = np.array(dist_lst)
# Indexes of matches where distance over threshold
thr_moved = 16.  
dist_over_thr_idx = np.where (dist_arr>thr_moved)[0]
# Pick top 20 indexes to draw
top_20_moved_matches_idx = dist_over_thr_idx[:50]

# Draw
top_20_moved_matches = [ matches[moved_match_id] for moved_match_id in top_20_moved_matches_idx ]
#img_matches_top_20_moved = cv.drawMatches(gray,kp,gray2,kp2,top_20_moved_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_matches_top_20_moved = cv.drawMatches(img,kp,img2,kp2,top_20_moved_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imwrite('D:/Startup/Visuals/SCO_Pics/SIFT/sift_kp_matches_moved.jpg',img_matches_top_20_moved)
