## Hidden features of a single image


Intermediate (hidden) layer activations of a single image using pre-trained VGG network.

### Original image:

![alt text](Visuals/ActivationsSingleImage/0/original.jpg "")

### Subsequent CNN, Pooling
Bellow are 3 images of subsequent layers: 2 convolutional + 1 maxpooling
(Note how the maxpooling layer "sharpens" the image (e.g. neuron in row 1, col 5))
![alt text](Visuals/ActivationsSingleImage/vgg_arch_3first.jpg "")

<img src="Visuals/ActivationsSingleImage/0/0.block1_conv1_0-64.jpg" width="280" height="280" /><img src="Visuals/ActivationsSingleImage/0/1.block1_conv2_0-64.jpg" width="280" height="280" /><img src="Visuals/ActivationsSingleImage/0/2.block1_pool_0-64.jpg" width="280" height="280" />

## Result of convolutions of the 5 CNN layers

Bellow is the result of the circumvented CNN layers of VGG
Note how in the beginning layers object can still be visually recognized by a human.
Note that from later layers one cannot easily identify the object. Later layers contain less information, those are extracted features. Some combination of the last layer features determine the class of the object. 




<img src="Visuals/ActivationsSingleImage/0/0.block1_conv1_0-64.jpg" width="280" height="280" /><img src="Visuals/ActivationsSingleImage/0/1.block1_conv2_0-64.jpg" width="280" height="280" /><img src="Visuals/ActivationsSingleImage/0/2.block1_pool_0-64.jpg" width="280" height="280" />  
<img src="Visuals/ActivationsSingleImage/0/3.block2_conv1_0-64.jpg" width="280" height="280" /><img src="Visuals/ActivationsSingleImage/0/4.block2_conv2_0-64.jpg" width="280" height="280" /><img src="Visuals/ActivationsSingleImage/0/5.block2_pool_0-64.jpg" width="280" height="280" />  
<img src="Visuals/ActivationsSingleImage/0/6.block3_conv1_0-64.jpg" width="280" height="280" /><img src="Visuals/ActivationsSingleImage/0/7.block3_conv2_0-64.jpg" width="280" height="280" /><img src="Visuals/ActivationsSingleImage/0/9.block3_pool_0-64.jpg" width="280" height="280" />  
<img src="Visualsidth="280" height="280" src="Visuals/Activati/ActivationsSingleImage/0/10.block4_conv1_0-64.jpg" width="280" height="280" /><img src="Visuals/ActivationsSingleImage/0/11.block4_conv2_0-64.jpg" width="280" height="280" /><img src="Visuals/ActivationsSingleImage/0/13.block4_pool_0-64.jpg" width="280" height="280" />  
<img src="Visuals/ActivationsSingleImage/0/14.block5_conv1_0-64.jpg" width="280" height="280" /><img src="Visuals/ActivationsSingleImage/0/15.block5_conv2_0-64.jpg" width="280" height="280" /><img src="Visuals/ActivationsSingleImage/0/17.block5_pool_0-64.jpg" width="280" height="280" />  

