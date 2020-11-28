import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
import time
from keras.applications.vgg16 import preprocess_input

class AugSequence (keras.utils.Sequence):

    def __init__(self, crop_range=1, allow_hor_flip=True, target_size=224, batch_size=32, \
        subtractMean = 0.0, pca_eigenvectors = None, pca_eigenvalues = None, preprocess="div255", \
        test=False, testtest=False, shuffle=True, datasrc="selfCreatedGoogle", debug=False): 
       
        self.target_size = target_size
        self.crop_range = crop_range
        self.allow_hor_flip = allow_hor_flip
        self.subtractMean = subtractMean
        self.pca_eigenvectors = pca_eigenvectors
        self.pca_eigenvalues = pca_eigenvalues
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
        #elif datasrc == "sco_initial":
        #    if test:
        #        data_dir = "D:\\Startup\\cs_images_20190509_val"
        #        #raise Exception('AugSequence: no test set defined for ' + datasrc)
        #    else:
        #        data_dir = "D:\\Startup\\cs_images_20190509_train"
        elif datasrc == "sco_v2":
            if test:
                data_dir = "C:\\TrainAndVal_SPTestLab\\Val"
                #raise Exception('AugSequence: no test set defined for ' + datasrc)
            else:
                data_dir = "C:\\TrainAndVal_SPTestLab\\Train"
        elif datasrc == "sco_v3":
            if testtest:
                data_dir = "C:\\RetellectImages\\Test"
            elif test:
                data_dir = "C:\\RetellectImages\\Val"
            else:
                data_dir = "C:\\RetellectImages\\Train"
        else:
            raise Exception('AugSequence: unknown datasrc')

        if preprocess=="div255":
            datagen = ImageDataGenerator ( rescale=1./255 )
        elif preprocess=="vgg":
            datagen = ImageDataGenerator ( preprocessing_function=preprocess_input )
        else:
            raise Exception('AugSequence: unknown preprocess parameter')
            
        
        size_uncropped = target_size + crop_range - 1

        self.data_generator = datagen.flow_from_directory(
            data_dir,
            target_size=(size_uncropped, size_uncropped),
            batch_size=batch_size,
            shuffle=shuffle,
            class_mode='categorical')

        #store length for faster retrieval of length
        self.len_value = len ( self.data_generator ) #* crop_range * crop_range

        # keep track how many items requested. Based on this counter, proper crop to be returned
        self.cnter = 0

        #initiate async thread for augmented data retrieval, which will be received via __getitem__()

    #Length of sequence is length of directory iterator for each crop variant
    def __len__( self ):
        return self.len_value

    def __getitem__( self, idx ): 
        #print ( "Starting getitem" )

        #get next uncropped batch of images
        X_uncropped, y = next ( self.data_generator )

        # get proper crop based on counter 
        start_w = np.random.randint ( 0, self.crop_range )
        start_h = np.random.randint ( 0, self.crop_range )
        horflip = np.random.choice(a=[False, True])
        if self.allow_hor_flip and horflip:
            X = np.flip ( X_uncropped [ : , start_w:start_w + self.target_size, start_h:start_h + self.target_size, : ] , axis = 1 )
        else:
            X = X_uncropped [ : , start_w:start_w + self.target_size, start_h:start_h + self.target_size, : ]

        #subtract to center values
        X -= self.subtractMean

        # PCA distortion
        if self.pca_eigenvectors is not None:
            # pca_eigenvectors is a 3x3 matrix; rows=R,G,B values; columns=Principal components orders by desc eigenvalues
            # pca_eigenvalues is a 3-array

            # for each principal component - random multiplier with std=1.0 and mean=0
            random_alpha = np.random.randn ( 3 ) * 1.0

            # Distort input values
            X += np.dot ( self.pca_eigenvectors, random_alpha * self.pca_eigenvalues)

        #update counter : max value is len of entire imageset 
        self.cnter += 1

        if self.debug and self.cnter%100 == 0:
            print ( "AugSequence.py, __getitem__, self.cnter, self.len_value:", str(self.cnter), " ", self.len_value, " ", time.strftime("%H:%M:%S") )
        #print ("X.shape, y.shape, start_w, start_h, target_size, self.cnter", X.shape, y.shape, start_w, start_h, self.target_size, self.cnter)

        return X, y

    def on_epoch_end(self):
        self.data_generator.reset()
        if self.debug:
            print ("AugSequence.py, on_epoch_end")

    def __del__(self):
        if self.debug:
            print ("AugSequence.py, __del__")

    def dataGen(self):
        return self.data_generator
