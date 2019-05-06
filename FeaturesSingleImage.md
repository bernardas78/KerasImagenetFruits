## Hidden features of a single image


Intermediate (hidden) layer activations of a single image using pre-trained VGG network.


### Network architecture

This bellow image is the VGG pre-trained network architecture.
![alt text](Visuals/ActivationsSingleImage/vgg_arch.jpg "")

### Original image:

![alt text](Visuals/ActivationsSingleImage/0/original.jpg "")

## Result of convolutions of the VGG network's CNN layers

- Bellow visualizations are result of convolutions of layers of pre-trained VGG of a single image. 
- There are 5 blocks in the VGG, so there are 5 rows in table of visualizations bellow. 
- In each row I draw the first 2 convolutions and pooling layer of the block. 
- Although blocks 3, 4, and 5 contain 3 convolutional layers, I skip them for resolution puproses, but you can find them in the source directory.
- I display up to 64 images per layer, although 2-5 blocks contain more filters (up to 512)

To note:
- In the beginning layers object can still be visually recognized by a human. 
- In later layers one cannot easily identify the object. Later layers contain less information, but it's more condensed (they become "features"). 
- Some combination of the last layer features determine the class of the object. 
 
- Note how the maxpooling layer "sharpens" the image (due to resolution, this can best be seen in last block)

| Blck | Convolution 1 | Convolution 2 | Max-Pool |
|:-----:|:-------------:|:-------------:|:--------:|
| 1 |<img src="Visuals/ActivationsSingleImage/0/0.block1_conv1_0-64.jpg" width="280" height="280" />|<img src="Visuals/ActivationsSingleImage/0/1.block1_conv2_0-64.jpg" width="280" height="280" />|<img src="Visuals/ActivationsSingleImage/0/2.block1_pool_0-64.jpg" width="280" height="280" />  |
| 2 |<img src="Visuals/ActivationsSingleImage/0/3.block2_conv1_0-64.jpg" width="280" height="280" />|<img src="Visuals/ActivationsSingleImage/0/4.block2_conv2_0-64.jpg" width="280" height="280" />|<img src="Visuals/ActivationsSingleImage/0/5.block2_pool_0-64.jpg" width="280" height="280" />  |
| 3 |<img src="Visuals/ActivationsSingleImage/0/6.block3_conv1_0-64.jpg" width="280" height="280" />|<img src="Visuals/ActivationsSingleImage/0/7.block3_conv2_0-64.jpg" width="280" height="280" />|<img src="Visuals/ActivationsSingleImage/0/9.block3_pool_0-64.jpg" width="280" height="280" />  |
| 4 |<img src="Visuals/ActivationsSingleImage/0/10.block4_conv1_0-64.jpg" width="280" height="280" />|<img src="Visuals/ActivationsSingleImage/0/11.block4_conv2_0-64.jpg" width="280" height="280" />|<img src="Visuals/ActivationsSingleImage/0/13.block4_pool_0-64.jpg" width="280" height="280" /> |
| 5 |<img src="Visuals/ActivationsSingleImage/0/14.block5_conv1_0-64.jpg" width="280" height="280" />|<img src="Visuals/ActivationsSingleImage/0/15.block5_conv2_0-64.jpg" width="280" height="280" />|<img src="Visuals/ActivationsSingleImage/0/17.block5_pool_0-64.jpg" width="280" height="280" />  |


