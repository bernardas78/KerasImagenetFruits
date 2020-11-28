#
# To run: 
#    cd C:\labs\KerasImagenetFruits\KerasTrainImagenet\Training_SCO
#    python adhoc_security_preLast__activations.py

import get_activations
from PIL import ImageFile, Image
import os
import cv2 as cv
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from keras.models import load_model

# Load model
print ("Loading model...")
#model = load_model("D:\Startup\models\model_v202_16classes_20200327.h5", custom_objects={'top_5': top_5})
model = load_model("J:\\AK Dropbox\\n20190113 A\\Models\\model_20200913_8prekes.h5")
print ("Loaded model")

# create single class one centroid
train_class_dir = r"C:\\RetelectImages\\Train\\991800900000" #agurkai

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

# euclidean distance of mahalanobis distances
def calcDist (mus, sigmas, activations,epsilon=0.000001):
    mahalanobis_dist = ( activations - mus ) / (sigmas+epsilon)
    return np.linalg.norm( mahalanobis_dist )

# -1 is last dense and -2 is droput; -3 is pre-last dense - what we need
pre_last_dense_layer_id = -3

# Extract pre-last dense layer
pre_last_dense_layer = model.layers[pre_last_dense_layer_id]

# Extract layer's output shape
pre_last_dense_out_shape = int(pre_last_dense_layer.output.shape[1]) # shape is (m,n), where m-number of samples; n-number neurons

# init array of all activations; shape [m,n], m - #samples; n - #neurons in pre-last layer
all_activations_preLast = np.empty((0,pre_last_dense_out_shape))
# collect activations of training images (Agurkai)
for file_name in os.listdir(train_class_dir):
    print ("Processing image " + file_name)
    img_preped = prepareImage ( "\\".join ( [train_class_dir,file_name] ) )
    imgs = np.stack ( [img_preped] ) #quick way to add a dimension
    img_activations_preLast = get_activations.get_layer_activations (model, imgs, pre_last_dense_layer)
    # add last image activations to all activations
    all_activations_preLast = np.vstack ( [ all_activations_preLast, img_activations_preLast])

#mus and sigmas for each neuron
mus = np.mean(all_activations_preLast, axis=0)
sigmas = np.std(all_activations_preLast, axis=0)

# Agurku distances from mean:
dist_agurkai = []
for sample_id in range(all_activations_preLast.shape[0]):
    sample_activations = all_activations_preLast[sample_id,:]
    sample_dist = calcDist ( mus=mus, sigmas=sigmas, activations=sample_activations)
    dist_agurkai.append(sample_dist)
print ("Agurkai, mean distance from Agurku center: {:4.0f}, std: {:4.0f}".format(np.mean(dist_agurkai), np.std(dist_agurkai)))

# collect activations of other class (Arbuzai)
all_activations_preLast_arbuzai = np.empty((0,pre_last_dense_out_shape))
for file_name in os.listdir("C:\\RetelectImages\\Train\\991806200000") : #Arbuzai
    print ("Processing image " + file_name)
    img_preped = prepareImage ( "\\".join ( ["C:\\RetelectImages\\Train\\991806200000",file_name] ) )
    imgs = np.stack ( [img_preped] ) #quick way to add a dimension
    img_activations_preLast = get_activations.get_layer_activations (model, imgs, pre_last_dense_layer)
    # add last image activations to all activations
    all_activations_preLast_arbuzai = np.vstack ( [ all_activations_preLast_arbuzai, img_activations_preLast])
# Arbuzai mus and sigmas for each neuron
#mus_arbuzai = np.mean(all_activations_preLast_arbuzai, axis=0)
#sigmas_arbuzai = np.std(all_activations_preLast_arbuzai, axis=0)


# Apelsinu distances from Agurku mean:
dist_arbuzai = []
for sample_id in range(all_activations_preLast_arbuzai.shape[0]):
    sample_activations = all_activations_preLast_arbuzai[sample_id,:]
    sample_dist = calcDist ( mus=mus, sigmas=sigmas, activations=sample_activations)
    dist_arbuzai.append(sample_dist)
print ("Arbuzai, distance from Agurkai mean: {:4.0f}, std: {:4.0f}".format(np.mean(dist_arbuzai), np.std(dist_arbuzai)))


# collect activations of Agurkai_validation
all_activations_preLast_agurkai_validation = np.empty((0,pre_last_dense_out_shape))
for file_name in os.listdir("C:\\RetelectImages\\Val\\991800900000"): #agurkai
    print ("Processing image " + file_name)
    img_preped = prepareImage ( "\\".join ( ["C:\\RetelectImages\\Val\\991800900000",file_name] ) )
    imgs = np.stack ( [img_preped] ) #quick way to add a dimension
    img_activations_preLast = get_activations.get_layer_activations (model, imgs, pre_last_dense_layer)
    # add last image activations to all activations
    all_activations_preLast_agurkai_validation = np.vstack ( [ all_activations_preLast_agurkai_validation, img_activations_preLast])

# Agurkai_validation distances from Agurku mean:
dist_agurkai_validation = []
for sample_id in range(all_activations_preLast_agurkai_validation.shape[0]):
    sample_activations = all_activations_preLast_agurkai_validation[sample_id,:]
    sample_dist = calcDist ( mus=mus, sigmas=sigmas, activations=sample_activations)
    dist_agurkai_validation.append(sample_dist)
print ("Agurkai_validation, distance from Agurkai mean: {:4.0f}, std: {:4.0f}".format(np.mean(dist_agurkai_validation), np.std(dist_agurkai_validation)))

# look at values
np.sort(dist_agurkai_validation)


# collect activations of not-Agurkai_validation
all_activations_preLast_notagurkai = np.empty((0,pre_last_dense_out_shape))
for file_name in os.listdir("C:\\RetelectImages\\NotAgurkai_Other"):
    print ("Processing image " + file_name)
    img_preped = prepareImage ( "\\".join ( ["C:\\RetelectImages\\NotAgurkai_Other",file_name] ) )
    imgs = np.stack ( [img_preped] ) #quick way to add a dimension
    img_activations_preLast = get_activations.get_layer_activations (model, imgs, pre_last_dense_layer)
    # add last image activations to all activations
    all_activations_preLast_notagurkai = np.vstack ( [ all_activations_preLast_notagurkai, img_activations_preLast])

# Agurkai_validation distances from Agurku mean:
dist_notagurkai_validation = []
for sample_id in range(all_activations_preLast_notagurkai.shape[0]):
    sample_activations = all_activations_preLast_notagurkai[sample_id,:]
    sample_dist = calcDist ( mus=mus, sigmas=sigmas, activations=sample_activations)
    dist_notagurkai_validation.append(sample_dist)
print ("Not_Agurkai_validation, distance from Agurkai mean: {:4.0f}, std: {:4.0f}".format(np.mean(dist_notagurkai_validation), np.std(dist_notagurkai_validation)))
# look at values
np.sort(dist_notagurkai_validation)

# Display ROC for agurkai vs. not agurkai; 
#   Positive class - not agurkai; TP - correct alert raised
y_score = dist_agurkai_validation + dist_notagurkai_validation
y_true = np.zeros(len(dist_agurkai_validation), dtype='int').tolist() + np.ones((len(dist_notagurkai_validation)), dtype='int').tolist()
(fpr,tpr,thresholds) = roc_curve (y_score=y_score, y_true=y_true)
roc_auc = auc(fpr, tpr)

plt.figure()
#lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Alertų %, kai nebandoma vogti')
plt.ylabel('Pagauta vagysčių, %')
plt.title('Žmogus pasirinko agurką')
plt.legend(loc="lower right")
plt.show()
