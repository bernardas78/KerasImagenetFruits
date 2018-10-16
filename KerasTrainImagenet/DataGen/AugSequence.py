import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile

class AugSequence (keras.utils.Sequence):

    def __init__(self, target_size=224, batch_size=32, test=False):          #(self, x_set, y_set, batch_size):
        #it used to throw file truncated error. bellow makes it tolerant to truncated files
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        if test:
            data_dir = "C:\\labs\\FruitDownload\\processed_split.imagenet\\validation"
        else:
            data_dir = "C:\\labs\\FruitDownload\\processed_split.imagenet\\train"

        datagen = ImageDataGenerator(rescale=1./255)

        self.data_generator = datagen.flow_from_directory(
            data_dir,
            target_size=(target_size, target_size),
            batch_size=batch_size,
            class_mode='categorical')

    def __len__( self ):
        return len ( self.data_generator )

    def __getitem__( self, idx ):
        X, y = next ( self.data_generator   )
        return X, y