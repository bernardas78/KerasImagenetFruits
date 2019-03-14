import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile, Image
import time
import pickle
import math
from keras.applications.vgg16 import preprocess_input

#import cv2
# Sample run:
#   dataGen = as_det_v7.AugSequence(batch_size=4, test=False, debug=True)
#   for (X,y) in dataGen:
#       print ("X.shape, y.shape:", X.shape, y.shape)
#       break
#   dataGen.getBatchFilenames()

class AugSequence (keras.utils.Sequence):

    def __init__(self, crop_range=1, target_size=224, batch_size=32, subdiv=3, \
        test=False, shuffle=True, datasrc="ilsvrc14_DET", debug=False):         
       
        #det_cat_hier_file = 'd:\ILSVRC14\det_cathier.obj'
        det_cat_desc_file = 'd:\ILSVRC14\det_catdesc.obj'

        self.target_size = target_size
        self.crop_range = crop_range
        self.debug = debug
        self.batch_size = batch_size
        self.subdiv = subdiv
        self.shuffle = shuffle

        #it used to throw file truncated error. bellow makes it tolerant to truncated files
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        if datasrc == "ilsvrc14_DET":
            if test:
                #self.data_dir = "D:\\ILSVRC14\\darknet_data\\JPEGImages\\ILSVRC2013_DET_val"
                # for speed copied to C:
                self.data_dir = "C:\\ILSVRC14\\ILSVRC2013_DET_val"
                img_filenames_file = "D:\\ILSVRC14\\det_img_val.obj"
                bboxes_file = "D:\\ILSVRC14\\det_bboxes_val.obj"
            else:
                #self.data_dir = "D:\\ILSVRC14\\darknet_data\\JPEGImages\\ILSVRC2014_DET_train_unp"
                # for speed copied to C:
                self.data_dir = "C:\\ILSVRC14\\ILSVRC2014_DET_train_unp"
                img_filenames_file = "D:\\ILSVRC14\\det_img_train.obj"
                bboxes_file = "D:\\ILSVRC14\\det_bboxes_train.obj"
        else:
            raise Exception('AugSequence: unknown datasrc')

        now=time.time()
        print ("Loading image file names from ", img_filenames_file)
        self.img_filenames = pickle.load( open(img_filenames_file, 'rb') )
        self.img_filenames_cnt = len(self.img_filenames)
        print ('Loaded in {0} seconds, total {1}'.format( time.time()-now, self.img_filenames_cnt ) )

        ## Extra images contain no bounding boxes (they are non-classes); eliminate them
        # Don't eliminate files without bounding boxes
        #non_class_indices = np.where ( np.char.find ( self.img_filenames, 'extra') < 0 )[0]
        #self.img_filenames = [ self.img_filenames[ind] for ind in non_class_indices
        
        now=time.time()
        print ("Loading bounding boxes from ", bboxes_file)
        self.bboxes = pickle.load( open(bboxes_file, 'rb') )
        print ('Loaded in {0} seconds, total {1}'.format( time.time()-now, len(self.bboxes) ) )

        #print ("Loading DET category hierarchy from ", det_cat_hier_file)
        #self.det_cat_hier = pickle.load( open(det_cat_hier_file, 'rb') )
        #print ("Finished loading DET category hierarchy")

        print ("Loading DET category names and indices from ", det_cat_desc_file)
        self.det_cats = pickle.load( open(det_cat_desc_file, 'rb') )
        print ("Finished loading DET category names and indices, total", len(self.det_cats) )

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

        #get next batch of images
        start_ind = np.min( [ self.cnter*self.batch_size, self.img_filenames_cnt ] )
        end_ind = np.min( [ (self.cnter+1)*self.batch_size, self.img_filenames_cnt ] )
        img_filesnames_batch = self.img_filenames [ start_ind : end_ind ]
        if self.debug:
            self.debug_filenames = img_filesnames_batch

        # Create X and y (extra class for non-class)
        #X = np.zeros ( (self.target_size,self.target_size,3,0 ) )
        X = np.zeros ( ( len(img_filesnames_batch), self.target_size, self.target_size, 3 ) )

        #y = np.zeros ( ( len(img_filesnames_batch), len(self.det_cats)+1 ) )
        # Last dimension of y is: Pr(obj), bx, by, bw, bh, P(class_1|obj), ..., P(class_n|obj)
        #y = np.zeros ( ( len(img_filesnames_batch), self.subdiv, self.subdiv, len(self.det_cats)+5 ) ) #DET_DEBUG
        #y = np.zeros ( ( len(img_filesnames_batch), self.subdiv, self.subdiv, 1 ) )
        y = np.zeros ( ( len(img_filesnames_batch), self.subdiv, self.subdiv, 5 ) )                     #DET_DEBUG
        #y = np.ones ( ( len(img_filesnames_batch), self.subdiv * self.subdiv * 4 ) ) 

        img_counter_in_batch = 0
        tm=np.zeros ((6))

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

            # Pre-process input (VGG)
            now=time.time()
            img_vgg_preprocessed = preprocess_input ( np.asarray(img_rgb) )
            tm[3]+=time.time()-now

            now=time.time()
            #img_arr = np.asarray ( img_rgb )
            X [ img_counter_in_batch, :, :, : ] = img_vgg_preprocessed
            tm[4]+=time.time()-now

            now=time.time()
            # create [ subdiv ^2 ] subdivisions, each containing Pr(obj), bx, by, bw, bh, P(class_1|obj), ..., P(class_n|obj)
            # Check if bounding box file exists
            if img_filename in self.bboxes.keys():
                # Multiple bboxes in each file possible. For this simple version, just overwrite if >1 in same subdivision
                for bbox in self.bboxes [ img_filename ]:
                    (class_label, xmin, xmax, ymin, ymax, img_width, img_height) = bbox
                    # Height and Width of a subdivision (different per image - used to determine bbox sizes relative to subdivisions)
                    subdiv_width = img_width / self.subdiv
                    subdiv_height = img_height / self.subdiv
                    # Determine absolute center of the bbox
                    bbox_x_center_abs = (xmax+xmin)/2
                    bbox_y_center_abs = (ymax+ymin)/2
                    # Subdivision of the bbox (if center of bbox belongs to subdivision)
                    bbox_x_ind_subdiv = int ( bbox_x_center_abs / subdiv_width )
                    bbox_y_ind_subdiv = int ( bbox_y_center_abs / subdiv_height )

                    # Size of bbox (relative to subdivision size)
                    #bbox_width_rel_subdiv = (xmax-xmin) / subdiv_width
                    #bbox_height_rel_subdiv = (ymax-ymin) / subdiv_height
                    # TEMPORARY - SO THAT SIGMOID WORKS Size of bbox (relative to image size)
                    bbox_width_rel_subdiv = (xmax-xmin) / img_width
                    bbox_height_rel_subdiv = (ymax-ymin) / img_height

                    # Center of bbox (relative to subdivision size)
                    bbox_x_center_subdiv = bbox_x_center_abs / subdiv_width - bbox_x_ind_subdiv
                    bbox_y_center_subdiv = bbox_y_center_abs / subdiv_height - bbox_y_ind_subdiv
                    # Assign y:
                    #   y[..., 0] = Pr (obj)
                    #   y[..., 1:4] = (bx, by, bw, bh)
                    #   y[..., class_index+5] = 1
                    try:
                        y [ img_counter_in_batch, bbox_x_ind_subdiv, bbox_y_ind_subdiv, 0 ] = 1.
                        y [ img_counter_in_batch, bbox_x_ind_subdiv, bbox_y_ind_subdiv, 1:5 ] = \
                            (bbox_x_center_subdiv, bbox_y_center_subdiv, bbox_width_rel_subdiv, bbox_height_rel_subdiv)
                        #DET_DEBUG
                        #y [ img_counter_in_batch, bbox_x_ind_subdiv*self.subdiv*self.subdiv + bbox_y_ind_subdiv*self.subdiv: \
                        #   bbox_x_ind_subdiv*self.subdiv*self.subdiv + bbox_y_ind_subdiv*self.subdiv + 4 ] = \
                        #    (bbox_x_center_subdiv, bbox_y_center_subdiv, bbox_width_rel_subdiv, bbox_height_rel_subdiv)
                        #y [ img_counter_in_batch, bbox_x_ind_subdiv, bbox_y_ind_subdiv, 0:4 ] = \
                        #    (bbox_x_center_subdiv, bbox_y_center_subdiv, bbox_width_rel_subdiv, bbox_height_rel_subdiv)
                        #class_index = self.det_cats [class_label] [0]
                        #y [ img_counter_in_batch, bbox_x_ind_subdiv, bbox_y_ind_subdiv, class_index+5] = 1.
                        #DET_DEBUG
                    except:
                        print ("Error creating y:", img_filename, bbox)
            tm[5]+=time.time()-now

            #if img_filename in self.bboxes.keys():
            #    bbox_label = self.bboxes [img_filename][0][0]               # str of bboxes: [('n000001' xmin xmax ymin ymax width height) () ...] 
            #    y [ img_counter_in_batch, self.det_cats[bbox_label][0] ] = 1.    # str of det_cats [ index description ]
            #else:
            #    y [ img_counter_in_batch, -1 ] = 1.                         # non-class when no bounding box found

            img_counter_in_batch += 1
        #print ("X.shape:", X.shape)
        if self.debug and self.cnter % 100 == 0:
            print ('Batch {0} in {1}'.format( self.cnter,  len(self) ), tm )

        #update counter : max value is len of entire imageset * crop_range^2
        self.cnter += 1

        return X, y

    def getBatchFilenames(self):
        return self.debug_filenames

    def on_epoch_end(self):
        if self.cnter >= len(self):
            self.cnter = 0

        #if self.shuffle:
        #    np.random.shuffle(self.img_filenames)
        #    print ("AugSequence_v7.py, Shuffling filenames")

        if self.debug:
            print ("AugSequence_v7.py, End of epoch")

    def __del__(self):
        if self.debug:
            print ("AugSequence_v7.py, __del__")
