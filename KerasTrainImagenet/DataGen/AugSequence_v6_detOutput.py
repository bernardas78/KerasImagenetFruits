import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile, Image
import time
import pickle
import math
#import cv2

class AugSequence (keras.utils.Sequence):

    def __init__(self, crop_range=1, target_size=224, batch_size=32, \
        test=False, shuffle=True, datasrc="ilsvrc14_DET", debug=False):         
       
        det_cat_hier_file = 'd:\ILSVRC14\det_cathier.obj'

        self.target_size = target_size
        self.crop_range = crop_range
        self.debug = debug
        self.batch_size = batch_size

        #it used to throw file truncated error. bellow makes it tolerant to truncated files
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        if datasrc == "ilsvrc14_DET":
            if test:
                self.data_dir = "C:\\ILSVRC14\\ILSVRC2013_DET_val"
                img_filenames_file = "D:\\ILSVRC14\\det_img_val.obj"
                bboxes_file = "D:\\ILSVRC14\\det_bboxes_val.obj"
            else:
                self.data_dir = "C:\\ILSVRC14\\ILSVRC2014_DET_train_unp"
                img_filenames_file = "D:\\ILSVRC14\\det_img_train.obj"
                bboxes_file = "D:\\ILSVRC14\\det_bboxes_train.obj"
        else:
            raise Exception('AugSequence: unknown datasrc')

        print ("Loading image file names from ", img_filenames_file)
        self.img_filenames = pickle.load( open(img_filenames_file, 'rb') )
        # Extra images contain no bounding boxes (they are non-classes)
        self.img_filenames = [ self.img_filenames[ind] for ind in np.where ( np.char.find ( self.img_filenames, 'extra') < 0 )[0] ]
        self.img_filenames_cnt = len(self.img_filenames)
        print ("Finished loading image file names")

        print ("Loading bounding boxes from ", bboxes_file)
        self.bboxes = pickle.load( open(bboxes_file, 'rb') )
        print ("Finished loading bounding boxes")

        print ("Loading DET category hierarchy from ", det_cat_hier_file)
        self.det_cat_hier = pickle.load( open(det_cat_hier_file, 'rb') )
        print ("Finished loading DET category hierarchy")

        # Shuffle the image files
        if shuffle:
            np.random.shuffle(self.img_filenames)

        #datagen = ImageDataGenerator(rescale=1./255)

        #size_uncropped = target_size + crop_range - 1

        #self.data_generator = datagen.flow_from_directory(
        #    data_dir,
        #    target_size=(size_uncropped, size_uncropped),
        #    batch_size=batch_size,
        #    class_mode='categorical')

        #store length for faster retrieval of length
        #self.len_value = len ( self.data_generator ) * crop_range * crop_range

        # keep track how many batches requested. Based on this counter, proper batch will be returned
        self.cnter = 0

        #initiate async thread for augmented data retrieval, which will be received via __getitem__()

    #Length of sequence is length of directory iterator for each crop variant
    def __len__( self ):
        return math.ceil( self.img_filenames_cnt / self.batch_size )

    def __getitem__( self, idx ): 
        #print ( "Starting getitem" )

        #get next uncropped batch of images
        start_ind = np.min( [ self.cnter*self.batch_size, self.img_filenames_cnt ] )
        end_ind = np.min( [ (self.cnter+1)*self.batch_size, self.img_filenames_cnt ] )
        img_filesnames_batch = self.img_filenames [ start_ind : end_ind ]

        #X = np.zeros ( (self.target_size,self.target_size,3,0 ) )
        X = np.zeros ( (self.target_size, self.target_size, 3, len(img_filesnames_batch) ) )
        img_counter_in_batch = 0

        tm=np.zeros ((4))
        # Read image data
        for img_filename in img_filesnames_batch:

            now=time.time()
            img = Image.open( '\\'.join ( [ self.data_dir, img_filename ] ) )
            #img = cv2.imread ( '\\'.join ( [ self.data_dir, img_filename ] ))
            tm[0]+=time.time()-now

            now=time.time()
            img_resized = img.resize ( (self.target_size,self.target_size) )
            #img_resized = cv2.resize(img, (self.target_size,self.target_size) )
            tm[1]+=time.time()-now

            now=time.time()
            img_rgb = img_resized.convert('RGB')
            tm[2]+=time.time()-now

            now=time.time()
            #img_arr = np.asarray ( img_rgb )
            X [ :, :, :, img_counter_in_batch ] = img_rgb
            tm[3]+=time.time()-now

            img_counter_in_batch += 1
        #print ("X.shape:", X.shape)
        if self.debug:
            print ('Batch {0} in {1}'.format( self.cnter,  len(self) ), tm )

        #update counter : max value is len of entire imageset * crop_range^2
        self.cnter += 1

        #return X, y
        return X

    def on_epoch_end(self):
        if self.cnter >= len(self):
            self.cnter = 0
        if self.debug:
            print ("AugSequence_v6.py, End of epoch")

    def __del__(self):
        if self.debug:
            print ("AugSequence_v6.py, __del__")
