# Prepares a data generator usable by Keras model training functions
#   Randomly flips images horizontally
#   Randomly shifts image verically/horizontally 10%
#   from imagenet fruit data of 20 classes
#
# To run: 
#   dataGen = dg_v2.prepDataGen()

from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile

def prepDataGen( test = False ):

    #it used to throw file truncated error. bellow makes it tolerant to truncated files
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    if test:
        data_dir = "C:\\labs\\FruitDownload\\processed_split.imagenet\\validation"
    else:
        data_dir = "C:\\labs\\FruitDownload\\processed_split.imagenet\\train"

    datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

    batch_size=32

    data_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')

    return data_generator