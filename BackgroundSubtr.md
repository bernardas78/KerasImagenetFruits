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

| Itm\Thr | Bread | Sour cream | Mushrooms |
|:-----:|:-------------:|:-------------:|:--------:|
| 800 <span style="color:red">**(too noisy around mushrooms)**</span> | <img src="Visuals/BackgroundSubtr/mask_duona_mog2_Thr800.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grietine_mog2_Thr800.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grybai_mog2_Thr800.png" width="298" height="192" />  |
| 1000 | <img src="Visuals/BackgroundSubtr/mask_duona_mog2_Thr1000.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grietine_mog2_Thr1000.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grybai_mog2_Thr1000.png" width="298" height="192" />  |
| 1200 <span style="color:red">**(bread becomes undetectable due to dark colors)**</span> | <img src="Visuals/BackgroundSubtr/mask_duona_mog2_Thr1200.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grietine_mog2_Thr1200.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grybai_mog2_Thr1200.png" width="298" height="192" />  |


**Knn**

| Itm\Thr | Bread | Sour cream | Mushrooms |
|:-----:|:-------------:|:-------------:|:--------:|
| 400 <span style="color:red">**(too noisy around mushrooms)**</span> | <img src="Visuals/BackgroundSubtr/mask_duona_knn_Thr400.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grietine_knn_Thr400.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grybai_knn_Thr400.png" width="298" height="192" />  |
| 800 <span style="color:red">**(still too noisy)**</span>| <img src="Visuals/BackgroundSubtr/mask_duona_knn_Thr800.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grietine_knn_Thr800.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grybai_knn_Thr800.png" width="298" height="192" />  |
| 1600 | <img src="Visuals/BackgroundSubtr/mask_duona_knn_Thr1600.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grietine_knn_Thr1600.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grybai_knn_Thr1600.png" width="298" height="192" />  |
| 2500 <span style="color:purple">**(dark colors start to fade in bread, but still ok)**</span>| <img src="Visuals/BackgroundSubtr/mask_duona_knn_Thr2500.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grietine_knn_Thr2500.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grybai_knn_Thr2500.png" width="298" height="192" />  |
| 4000 <span style="color:green">**(Best!)**</span> | <img src="Visuals/BackgroundSubtr/mask_duona_knn_Thr4000.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grietine_knn_Thr4000.png" width="298" height="192" />|<img src="Visuals/BackgroundSubtr/mask_grybai_knn_Thr4000.png" width="298" height="192" />  |


### Using filters to remove small patches of mask
- Problem: background subtractors usually detect small patches outside of area that really changed (first image)
- In order to localize items, those small patches should be excluded
- Applying convoltional filter of [ k x k ] size to ensure at least half of pixels are classified as foreground

| Filter | Bread |
|:-----:|:-------------:|
| Identity (i.e. no filter) | <img src="Visuals/BackgroundSubtr/Orig/filter_identity_circum.png" width="298" height="192" />|
| 2x2, min 2 pixels | <img src="Visuals/BackgroundSubtr/filter_2by2_2min.png" width="298" height="192" />|
| 3x3, min 5 pixels | <img src="Visuals/BackgroundSubtr/filter_3by3_5min.png" width="298" height="192" />|
| 4x4, min 8 pixels | <img src="Visuals/BackgroundSubtr/filter_4by4_8min.png" width="298" height="192" />|
| 5x5, min 13 pixels <span style="color:green">**(Best!)**</span> | <img src="Visuals/BackgroundSubtr/filter_5by5_13min.png" width="298" height="192" />|


## Conclusions
- For background subtraction KNN better then MOG2
- Using convolutional filters for removing accidentally detected small areas works