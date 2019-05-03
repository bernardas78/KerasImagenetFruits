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

Convolution 1 | Convolution 2 | Convolution 3 | Max pooling 
--- | --- | --- 
<img src="Visuals/ActivationsSingleImage/0/0.block1_conv1_0-64.jpg" width="200" height="200" /> | <img src="Visuals/ActivationsSingleImage/0/0.block1_conv1_0-64.jpg" width="200" height="200" /> |  | <img src="Visuals/ActivationsSingleImage/0/0.block1_conv1_0-64.jpg" width="200" height="200" /> 
<img src="Visuals/ActivationsSingleImage/0/0.block1_conv1_0-64.jpg" width="200" height="200" /> | <img src="Visuals/ActivationsSingleImage/0/0.block1_conv1_0-64.jpg" width="200" height="200" /> |  | <img src="Visuals/ActivationsSingleImage/0/0.block1_conv1_0-64.jpg" width="200" height="200" /> 
<img src="Visuals/ActivationsSingleImage/0/0.block1_conv1_0-64.jpg" width="200" height="200" /> | <img src="Visuals/ActivationsSingleImage/0/0.block1_conv1_0-64.jpg" width="200" height="200" /> |  | <img src="Visuals/ActivationsSingleImage/0/0.block1_conv1_0-64.jpg" width="200" height="200" /> 
<img src="Visuals/ActivationsSingleImage/0/0.block1_conv1_0-64.jpg" width="200" height="200" /> | <img src="Visuals/ActivationsSingleImage/0/0.block1_conv1_0-64.jpg" width="200" height="200" /> |  | <img src="Visuals/ActivationsSingleImage/0/0.block1_conv1_0-64.jpg" width="200" height="200" /> 
<img src="Visuals/ActivationsSingleImage/0/0.block1_conv1_0-64.jpg" width="200" height="200" /> | <img src="Visuals/ActivationsSingleImage/0/0.block1_conv1_0-64.jpg" width="200" height="200" /> |  | <img src="Visuals/ActivationsSingleImage/0/0.block1_conv1_0-64.jpg" width="200" height="200" /> 



