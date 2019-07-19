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

Analyzed 3 situations when items were placed:

| Item | Before | After |
|:-----:|:-------------:|:--------:|
| Bread | <img src="Visuals/BackgroundSubtr/BeforeDuona.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/AfterDuona.png" width="298" height="192" /> |
| Sour cream | <img src="Visuals/BackgroundSubtr/BeforeGrietine.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/AfterGrietine.png" width="298" height="192" /> |
| Mushrooms | <img src="Visuals/BackgroundSubtr/BeforeGrybai.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/AfterGrybai.png" width="298" height="192" /> |


**Mog2**

| Itm\Thr | 800 | 1000 | 1200 |
|:-----:|:-------------:|:-------------:|:--------:|
| Bread | <img src="Visuals/BackgroundSubtr/mask_duona_mog2_Thr800.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_duona_mog2_Thr1000.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_duona_mog2_Thr1200.png" width="298" height="192" />  |
| Sour cream | <img src="Visuals/BackgroundSubtr/mask_grietine_mog2_Thr800.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grietine_mog2_Thr1000.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grietine_mog2_Thr1200.png" width="298" height="192" />  |
| Mushrooms | <img src="Visuals/BackgroundSubtr/mask_grybai_mog2_Thr800.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grybai_mog2_Thr1000.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grybai_mog2_Thr1200.png" width="298" height="192" />  |

**Mog2**

| Itm\Thr | Bread | Sour cream | Mushrooms |
|:-----:|:-------------:|:-------------:|:--------:|
| 800 | <img src="Visuals/BackgroundSubtr/mask_duona_mog2_Thr800.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grietine_mog2_Thr800.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grybai_mog2_Thr800.png" width="298" height="192" />  |
| 1000 | <img src="Visuals/BackgroundSubtr/mask_duona_mog2_Thr1000.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grietine_mog2_Thr1000.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grybai_mog2_Thr1000.png" width="298" height="192" />  |
| 1200 | <img src="Visuals/BackgroundSubtr/mask_duona_mog2_Thr1200.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grietine_mog2_Thr1200.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grybai_mog2_Thr1200.png" width="298" height="192" />  |


**Knn**

| Itm\Thr | Bread | Sour cream | Mushrooms |
|:-----:|:-------------:|:-------------:|:--------:|
| 400 | <img src="Visuals/BackgroundSubtr/mask_duona_knn_Thr400.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grietine_knn_Thr400.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grybai_knn_Thr400.png" width="298" height="192" />  |
| 800 | <img src="Visuals/BackgroundSubtr/mask_duona_knn_Thr800.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grietine_knn_Thr800.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grybai_knn_Thr800.png" width="298" height="192" />  |
| 1600 | <img src="Visuals/BackgroundSubtr/mask_duona_knn_Thr1600.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grietine_knn_Thr1600.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grybai_knn_Thr1600.png" width="298" height="192" />  |
| 2500 | <img src="Visuals/BackgroundSubtr/mask_duona_knn_Thr2500.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grietine_knn_Thr2500.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grybai_knn_Thr2500.png" width="298" height="192" />  |
| 4000 | <img src="Visuals/BackgroundSubtr/mask_duona_knn_Thr4000.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grietine_knn_Thr4000.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grybai_knn_Thr4000.png" width="298" height="192" />  |
