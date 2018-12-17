# Prepares a data generator usable by Keras model training functions
#   from imagenet fruit data of 20 classes
#
# To run: 
#   dataGen = dg_v1.prepDataGen()

from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
#from keras.applications.vgg16 import preprocess_input
#import numpy as np

def prepDataGen( target_size=150, test = False, batch_size = 64, datasrc="selfCreatedGoogle" ):

    #it used to throw file truncated error. bellow makes it tolerant to truncated files
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    if datasrc == "selfCreatedGoogle":
        if test:
            data_dir = "C:\\labs\\FruitDownload\\processed_split.imagenet\\validation"
        else:
            data_dir = "C:\\labs\\FruitDownload\\processed_split.imagenet\\train"
    elif datasrc == "ilsvrc14":
        if test:
            data_dir = "C:\\ILSVRC14\\ILSVRC2012_img_val_unp_20"
        else:
            data_dir = "C:\\ILSVRC14\\ILSVRC2012_img_train_unp_20"
    elif datasrc == "ilsvrc14_50classes":
        if test:
            data_dir = "C:\\ILSVRC14\\ILSVRC2012_img_val_unp_50"
        else:
            data_dir = "C:\\ILSVRC14\\ILSVRC2012_img_train_unp_50"
    elif datasrc == "ilsvrc14_100classes":
        if test:
            data_dir = "C:\\ILSVRC14\\ILSVRC2012_img_val_unp_100"
        else:
            data_dir = "C:\\ILSVRC14\\ILSVRC2012_img_train_unp_100"
    elif datasrc == "ilsvrc14_100boundingBoxes":
        if test:
            data_dir = "C:\\ILSVRC14\\ILSVRC2012_img_val_unp_100"
        else:
            data_dir = "C:\\ILSVRC14\\ILSVRC2012_img_train_bbox_100"
    elif datasrc == "ilsvrc14_full":
        if test:
            data_dir = "D:\\ILSVRC14\\ILSVRC2012_img_val_unp"
        else:
            data_dir = "D:\\ILSVRC14\\ILSVRC2012_img_train_unp"
    else:
        raise Exception('AugSequence: unknown datasrc')

    #def customAug(x):
    #    #x[:,:,0] -= 103.939 #/255
    #    #x[:,:,1] -= 116.779 #/255
    #    #x[:,:,2] -= 123.68 #/255
    #    #x = ( (x-np.min(x)) / (np.max(x)-np.min(x)) ) * 2. - 1.
    #    x[:,:,0] = ( (x[:,:,0]-np.min(x[:,:,0])) / (np.max(x[:,:,0])-np.min(x[:,:,0])) ) * 2. - 1.
    #    x[:,:,1] = ( (x[:,:,1]-np.min(x[:,:,1])) / (np.max(x[:,:,1])-np.min(x[:,:,1])) ) * 2. - 1.
    #    x[:,:,2] = ( (x[:,:,2]-np.min(x[:,:,2])) / (np.max(x[:,:,2])-np.min(x[:,:,2])) ) * 2. - 1.
    #    #print ("x.shape:",x.shape)

    #    return x

    datagen = ImageDataGenerator ( rescale=1./255 )
    #datagen = ImageDataGenerator ( rescale=1./255, preprocessing_function=customAug )
    #datagen = ImageDataGenerator ( rescale=1., preprocessing_function=preprocess_input )

    

    #batch_size=32
    print ("DataGen_v1_150x150_1frame.py: Batch size:", str(batch_size) )

    data_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(target_size, target_size),
        batch_size=batch_size,
        class_mode='categorical')

    return data_generator