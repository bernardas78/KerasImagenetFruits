
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
import h5py

def VGG_16(weights_path=None):
    model = Sequential()
    
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
    #model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        #model.load_weights(weights_path)
        f = h5py.File(weights_path)
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers) - 1:
                # we don't look at the last two layers in the savefile (fully-connected and activation)
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            layer = model.layers[k]

            print ("Layer",layer.__class__.__name__)
            if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D', 'Conv2D']:
                weights[0] = np.transpose(weights[0], (2, 3, 1, 0))

            layer.set_weights(weights)

        f.close()

    return model

# import keras.backend as K
# K.set_image_data_format('channels_first')
# K.image_data_format()

# exec(open("vgg-16_keras.py").read())
# model = VGG_16('D:\\ILSVRC14\\models\\vgg16_weights.h5')
# target_size=224
# datasrc="ilsvrc14_full"
# subtractMean = np.array([103.939, 116.779, 123.68]) /256.
# e_v2.eval(model, target_size=target_size, datasrc=datasrc, subtractMean=subtractMean, test=True)
# model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
# e_v1.eval ( model, target_size=target_size, datasrc=datasrc, eval_test=True, eval_train=False )


from keras.preprocessing.image import load_img
#from keras.preprocessing.image import reshape
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
model = VGG16()




# load an image from file
image = load_img('D:\\ILSVRC14\\ILSVRC2012_img_val_unp\\n03063599\\ILSVRC2012_val_00034882.jpeg', target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# predict the probability across all output classes
yhat = model.predict(image)
# convert the probabilities to class labels
label = decode_predictions(yhat)
# retrieve the most likely result, e.g. highest probability
label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))

#if __name__ == "__main__":
#    im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
#    im[:,:,0] -= 103.939
#    im[:,:,1] -= 116.779
#    im[:,:,2] -= 123.68
#    im = im.transpose((2,0,1))
#    im = np.expand_dims(im, axis=0)

#    # Test pretrained model
#    model = VGG_16('vgg16_weights.h5')
#    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#    model.compile(optimizer=sgd, loss='categorical_crossentropy')
#    out = model.predict(im)
#    print (np.argmax(out))