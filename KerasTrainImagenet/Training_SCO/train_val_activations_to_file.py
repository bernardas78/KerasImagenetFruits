# Activations of [train and val] set, pre-last layer to files
#
# To run: 
#    cd C:\labs\KerasImagenetFruits\KerasTrainImagenet\Training_SCO
#    python train_val_activations_to_file.py

from PIL import ImageFile, Image
import os
import cv2 as cv
import numpy as np
from keras.backend import function
from keras.models import load_model
from keras.metrics import top_k_categorical_accuracy
import pickle


# remove this later (needed to load a model
#def top_5(y_true, y_pred):
#    return top_k_categorical_accuracy(y_true, y_pred, k=5)

# A proxy to get givel layer's activations of a given model
def get_layer_activations(model, X, layer):
    # Initialize a return value: list of activations
    #layer = model.layers[layer_id]
    func_activation = function([model.input], [layer.output])
    output_activation = func_activation([X])[0]
    return output_activation

# Paveikslelio paruosimo f-ja (vgg)
#def prepareImage (filename):
#	# input duomenu dydis (modelio savybe)
#	target_size = 224
#	# Load image ( BGR )
#	img = cv.imread(filename)
#	# Resize to target
#	img = cv.resize ( img, (target_size,target_size) )
#	# Subtract global dataset average to center pixels to 0
#	img = img - [103.939, 116.779, 123.68]
#	return img

# Paveikslelio paruosimo f-ja (self-created model)
def prepareImage (filename):
	# input duomenu dydis (modelio savybe)
	target_size = 150
	# Load image ( BGR )
	img = cv.imread(filename)
	# convert to RGB
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	# Resize to target
	img = cv.resize ( img, (target_size,target_size) )
	# Subtract global dataset average to center pixels to 0
	img = img / 255.
	return img

# Processes single set (train or val)
def process_train_or_val (folder, model, activations_filename):

    # Extract pre-last dense layer
    pre_last_layer = model.layers[pre_last_layer_id]

    # Extract layer's output shape
    out_shape = int(pre_last_layer.output.shape[1]) # shape is (m,n), where m-number of samples; n-number neurons

    # init array of all activations; shape [m,n], m - #samples; n - #neurons in pre-last layer
    all_activations_preLast = np.empty((0,out_shape))

    # collect activations of images
    i=0
    for _,class_dirs,_ in os.walk(folder):
        for class_dir in class_dirs:
            print("Current class: ",class_dir,folder)
            for file_name in os.listdir("\\".join([folder , class_dir]) ):
                i+=1
                if i%20==0:
                    print ("Processed {0} images ".format(i) )
                img_preped = prepareImage ( "\\".join ( [folder,class_dir,file_name] ) )
                imgs = np.stack ( [img_preped] ) #quick way to add a dimension
                img_activations_preLast = get_layer_activations (model, imgs, pre_last_layer)
                # add last image activations to all activations
                all_activations_preLast = np.vstack ( [ all_activations_preLast, img_activations_preLast])
    print ("Shape all_activations_preLast:", all_activations_preLast.shape)
    print ("All activations min:{}, max:{}, mean:{}".format(np.min(all_activations_preLast), np.max(all_activations_preLast), np.mean(all_activations_preLast)))

    #Save activations to file
    print ( "Save activations to file" )
    with open(activations_filename, 'wb') as file_desc:
        pickle.dump(all_activations_preLast, file_desc)
    return

# Train data used later to make classifiers
#train_dir = r"C:\TrainAndVal\Train"
#val_dir = r"C:\TrainAndVal\Val"
train_dir = r"C:\\RetelectImages\\Train"
val_dir = r"C:\\RetelectImages\\Val"

# set manually pre-last layer: -2 if nothing after pre-last dense is used
pre_last_layer_id = -3


# Load model
print ("Loading model...")
#model = load_model("D:\Startup\models\model_v202_16classes_20200327.h5", custom_objects={'top_5': top_5})
model = load_model("J:\\AK Dropbox\\n20190113 A\\Models\\model_20200913_8prekes.h5")
print ("Loaded model")
#model = load_model("D:\ILSVRC14\models\model_v55.h5", custom_objects={'top_5': m_v10.top_5})

#process_train_or_val(train_dir)
process_train_or_val(folder=train_dir, model=model, activations_filename = "D:\\Startup\\Activations_preLast_Security\\act_preLast_train.obj")
process_train_or_val(folder=val_dir, model=model, activations_filename = "D:\\Startup\\Activations_preLast_Security\\act_preLast_val.obj")
#all_activations_preLast_val = pickle.load( open("D:\\Startup\\Activations_preLast_Security\\act_preLast_val.obj", 'rb') )
#all_activations_preLast_train = pickle.load( open("D:\\Startup\\Activations_preLast_Security\\act_preLast_train.obj", 'rb') )
