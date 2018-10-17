import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
import time

class AugSequence (keras.utils.Sequence):

    def __init__(self, crop_range=1, target_size=224, batch_size=32, test=False, debug=False):          #(self, x_set, y_set, batch_size):
       
        self.target_size = target_size
        self.crop_range = crop_range
        self.debug = debug

        #it used to throw file truncated error. bellow makes it tolerant to truncated files
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        if test:
            data_dir = "C:\\labs\\FruitDownload\\processed_split.imagenet\\validation"
        else:
            data_dir = "C:\\labs\\FruitDownload\\processed_split.imagenet\\train"

        datagen = ImageDataGenerator(rescale=1./255)

        size_uncropped = target_size + crop_range - 1

        self.data_generator = datagen.flow_from_directory(
            data_dir,
            target_size=(size_uncropped, size_uncropped),
            batch_size=batch_size,
            class_mode='categorical')

        #store length for faster retrieval of length
        self.len_value = len ( self.data_generator ) * crop_range * crop_range

        # keep track how many items requested. Based on this counter, proper crop to be returned
        self.cnter = 0

    #Length of sequence is length of directory iterator for each crop variant
    def __len__( self ):
        return self.len_value

    def __getitem__( self, idx ): 
        #print ( "Starting getitem" )

        #get next uncropped batch of images
        X_uncropped, y = next ( self.data_generator )

        # get proper crop based on counter 
        counter_epoch = int ( self.cnter / len ( self.data_generator ) )
        start_w = int ( counter_epoch / self.crop_range )
        start_h = counter_epoch % self.crop_range 
        X = X_uncropped [ : , start_w:start_w + self.target_size, start_h:start_h + self.target_size, : ]

        #update counter : max value is len of entire imageset * crop_range^2
        self.cnter += 1
        if self.cnter >= self.len_value:
            self.cnter = 0

        #shuffle data when starting a new epoch
        if self.cnter % len ( self.data_generator ) == 0:
            #print ("Resetting data generator")
            self.data_generator.reset()

        if self.debug and self.cnter%100 == 0:
            print ( "AugSequence.py, __getitem__, self.cnter, self.len_value:", str(self.cnter), " ", self.len_value, " ", time.strftime("%H:%M:%S") )
        #print ("X.shape, y.shape, start_w, start_h, target_size, self.cnter", X.shape, y.shape, start_w, start_h, self.target_size, self.cnter)
        return X, y

    def on_epoch_end(self):
        print ("End of epoch")