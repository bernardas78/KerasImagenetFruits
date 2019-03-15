# Multiple item detection for self-checkout and regular Point-of-sale


## Problem definition
- Retailers usually have a purchased items bagging area where goods are placed after scanning
- Sometimes unscanned (candidates for stealing) are placed to that area

## Proposed solution
- Install a camera to take a photo of bagging area
- Detect items in bagging area
- Alert store employee if unrecognized items found which were not purchased


## Dataset
- 200 classes 
- Data from Imagenet ILSVRC14 DET Challenge

## Techniques used:
- YoloV3 (download pre-trained)
- Train for this custom dataset

## Visuals received:
More pictures [here.](DetPredictions.md)

![alt text](Visuals/PredYolo/ILSVRC2012_val_00000006.JPEG "")
![alt text](Visuals/PredYolo/ILSVRC2012_val_00000007.JPEG "")
![alt text](Visuals/PredYolo/ILSVRC2012_val_00000018.JPEG "")
