import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
import time
import threading

class AugSequence (keras.utils.Sequence):

    def __init__(self, crop_range=1, target_size=224, batch_size=32, test=False, datasrc="selfCreatedGoogle", debug=False):          #(self, x_set, y_set, batch_size):
       
        self.target_size = target_size
        self.crop_range = crop_range
        self.debug = debug

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
        else:
            raise Exception('AugSequence: unknown datasrc')

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

        #initiate async thread for augmented data retrieval, which will be received via __getitem__()
        self.pdaThread = threading.Thread(target=self.prepDataAsync)
        self.pdaThread.start()

    #Length of sequence is length of directory iterator for each crop variant
    def __len__( self ):
        return self.len_value

    def __getitem__( self, idx ): 
        #print ( "Starting getitem" )

        #join async thread of augmented data retrieval
        now=time.perf_counter()
        self.pdaThread.join()
        print ("self.pdaThread.join() ...",time.perf_counter()-now)
        del self.pdaThread

        #get next uncropped batch of images
        #now=time.perf_counter()
        #X_uncropped, y = next ( self.data_generator )
        X_uncropped, y =  self.X_uncropped, self.y
        #print ("X_uncropped, y  ...",time.perf_counter()-now)

        # get proper crop based on counter 
        #now=time.perf_counter()
        counter_epoch = int ( self.cnter / len ( self.data_generator ) )
        start_w = int ( counter_epoch / self.crop_range )
        start_h = counter_epoch % self.crop_range 
        #print ("3 math commands ...",time.perf_counter()-now)

        #now=time.perf_counter()
        X = X_uncropped [ : , start_w:start_w + self.target_size, start_h:start_h + self.target_size, : ]
        #print ("X = X_uncropped ...",time.perf_counter()-now)

        #update counter : max value is len of entire imageset * crop_range^2
        #now=time.perf_counter()
        self.cnter += 1
        if self.cnter >= self.len_value:
            self.cnter = 0
        #print ("if self.cnter >=...",time.perf_counter()-now)

        #shuffle data when starting a new epoch
        #now=time.perf_counter()
        if self.cnter % len ( self.data_generator ) == 0:
            #print ("Resetting data generator")
            self.data_generator.reset()
        #print ("self.data_gener ...",time.perf_counter()-now)

        if self.debug and self.cnter%100 == 0:
            print ( "AugSequence.py, __getitem__, self.cnter, self.len_value:", str(self.cnter), " ", self.len_value, " ", time.strftime("%H:%M:%S") )
        #print ("X.shape, y.shape, start_w, start_h, target_size, self.cnter", X.shape, y.shape, start_w, start_h, self.target_size, self.cnter)

        #initiate async thread for augmented data retrieval, which will be received via next call __getitem__()
        self.pdaThread = threading.Thread(target=self.prepDataAsync)
        self.pdaThread.start()

        return X, y

    def prepDataAsync(self):
        self.X_uncropped, self.y = next ( self.data_generator )

    def on_epoch_end(self):
        print ("End of epoch")

    def __del__(self):
        print ("AugSequence.py, __del__ started")
        #join and kill last thread
        self.pdaThread.join()
        del self.pdaThread
        print ("AugSequence.py, __del__ finished")
