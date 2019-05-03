## Hidden features of a single image

Intermediate (hidden) layer activations of a single image using pre-trained VGG network.

### Original image:

![alt text](Visuals/ActivationsSingleImage/0/original.jpg "")

### Subsequent CNN, Pooling
Bellow are 3 images of subsequent layers: 2 convolutional + 1 maxpooling
(Note how the maxpooling layer "sharpens" the image (e.g. neuron in row 1, col 5))
![alt text](Visuals/ActivationsSingleImage/vgg_arch_3first.jpg "")

Convolution 1:&nbsp;
![alt text](Visuals/ActivationsSingleImage/0/0.block1_conv1_0-64.jpg "") 
Convolution 2:&nbsp;
![alt text](Visuals/ActivationsSingleImage/0/1.block1_conv2_0-64.jpg "")
Maxpooling:&nbsp;
![alt text](Visuals/ActivationsSingleImage/0/2.block1_pool_0-64.jpg "")

## Result of convolutions of the 5 CNN layers

Bellow is the result of the circumvented CNN layers of VGG
![alt text](Visuals/ActivationsSingleImage/vgg_arch_5cnn.jpg "")

![alt text](Visuals/ActivationsSingleImage/0/0.block1_conv1_0-64.jpg "")
![alt text](Visuals/ActivationsSingleImage/0/3.block2_conv1_0-64.jpg "")
![alt text](Visuals/ActivationsSingleImage/0/6.block3_conv1_0-64.jpg "")
![alt text](Visuals/ActivationsSingleImage/0/10.block4_conv1_0-64.jpg "")
![alt text](Visuals/ActivationsSingleImage/0/14.block5_conv1_0-64.jpg "")
