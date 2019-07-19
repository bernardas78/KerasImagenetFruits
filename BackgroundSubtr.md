# Background Subtraction

## Problem definition
- Customer places items 1 by 1 in bagging area
- Sometimes items not scanned are placed into bagging area with purpose to steal them
- Upon encountering a new item in bagging area which was not scanned - an alert may be generated to a store attendant
- Purpose of this task is to detect an d localize new item in bagging area

## Techniques used
- Motion detection is done using a simple KNN background subtractor, comparing a previous frame with current
- After motion stops (presumably, customer moved his arm away from bagging area and left the item there):
  - MOG2 bacground subtractor ( *cv.createBackgroundSubtractorMOG2()* )
  - KNN background subtractor ( *cv.createBackgroundSubtractorKNN()* )

## Results


**Mog2**

| Threshold | Bread | Sour cream | Mushrooms |
|:-----:|:-------------:|:-------------:|:--------:|
| 800 | <img src="Visuals/BackgroundSubtr/mask_duona_mog2_Thr800.png" width="280" height="280" />|<img src="Visuals/BackgroundSubtr/mask_grietine_mog2_Thr800.png" width="280" height="280" />|<img src="Visuals/BackgroundSubtr/mask_grybai_mog2_Thr800.png" width="280" height="280" />  |
| 1000 | <img src="Visuals/BackgroundSubtr/mask_duona_mog2_Thr1000.png" width="280" height="280" />|<img src="Visuals/BackgroundSubtr/mask_grietine_mog2_Thr1000.png" width="280" height="280" />|<img src="Visuals/BackgroundSubtr/mask_grybai_mog2_Thr1000.png" width="280" height="280" />  |
| 1200 | <img src="Visuals/BackgroundSubtr/mask_duona_mog2_Thr1200.png" width="280" height="280" />|<img src="Visuals/BackgroundSubtr/mask_grietine_mog2_Thr1200.png" width="280" height="280" />|<img src="Visuals/BackgroundSubtr/mask_grybai_mog2_Thr1200.png" width="280" height="280" />  |
