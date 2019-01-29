# (1) Prepares labels files for darknet: outputs 1 txt file per image (only output txt files for images with bboxes)
# (2) Prepares a list of image filenames (only ones with bboxes in them).
# (3) List of category names
#   Source: bboxes in single file (result of bboxesToListToFile.py)
#           Detection categories file (result of imagenetCatHierToDictToFile.py)
#   (1) Destination: XML files in D:\ILSVRC14\darknet_data\labels\ILSVRC2014_DET_train_unp and
#                             D:\ILSVRC14\darknet_data\labels\ILSVRC2013_DET_val\all
#   (2) Destination: D:\ILSVRC14\darknet_data\train.txt and 
#                    D:\ILSVRC14\darknet_data\val.txt
#   (3) Destination: D:\ILSVRC14\darknet_data\imagenet.names
# To Run:
#   cd C:\labs\KerasImagenetFruits\PreprocessImages
#   python
#   Set train or validation  
Test = False
#   exec(open("labelPrepForDarknet.py").read())

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

#sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

#classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

#img_path = 'D:\ILSVRC14\darknet_data\JPEGImages\ILSVRC2014_DET_train_unp'

# Categories
det_cats = pickle.load( open('d:\ILSVRC14\det_catdesc.obj', 'rb') )

# Write detection categories into a file understandable by darknet
darknet_categories_filename = "D:\\ILSVRC14\\darknet_data\\imagenet.names"
darknet_categories_file = open( darknet_categories_filename, 'w')
#_ = [ darknet_categories_file.write ( cat_name + '\n' ) for cat_name in det_cats.keys() ]  #nXXXXXXX
_ = [ darknet_categories_file.write ( det_cats[cat_name][1] + '\n' ) for cat_name in det_cats.keys() ]   #human readable labels
darknet_categories_file.close()

if not Test:
    darknet_labels_path = "D:\\ILSVRC14\\darknet_data\\labels\\ILSVRC2014_DET_train_unp"
    bboxes_file = "d:\\ILSVRC14\\det_bboxes_train.obj"
    images_root = "D:\\ILSVRC14\\darknet_data\\JPEGImages\\ILSVRC2014_DET_train_unp"
    imagefilenames_filename = "D:\\ILSVRC14\\darknet_data\\train.txt"
else:
    darknet_labels_path = "D:\\ILSVRC14\\darknet_data\\labels\\ILSVRC2013_DET_val"
    bboxes_file = "d:\\ILSVRC14\\det_bboxes_val.obj"
    images_root = "D:\\ILSVRC14\\darknet_data\\JPEGImages\\ILSVRC2013_DET_val"
    imagefilenames_filename = "D:\\ILSVRC14\\darknet_data\\val.txt"

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

#def convert_annotation(year, image_id):
#    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
#    out_file = open('VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
#    tree=ET.parse(in_file)
#    root = tree.getroot()
#    size = root.find('size')
#    w = int(size.find('width').text)
#    h = int(size.find('height').text)

#    for obj in root.iter('object'):
#        difficult = obj.find('difficult').text
#        cls = obj.find('name').text
#        if cls not in classes or int(difficult) == 1:
#            continue
#        cls_id = classes.index(cls)
#        xmlbox = obj.find('bndbox')
#        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
#        bb = convert((w,h), b)
#        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

#wd = getcwd()

# Create a single image-filenames file
imagefilenames_file = open( imagefilenames_filename, 'w')

# Load bboxes file
print ("Loading bounding boxes...")
bboxes = pickle.load( open(bboxes_file, 'rb') )
print ("Finished loading bounding boxes")

# For each image file (multiple bboxes possible in single image)
for single_img_filename in bboxes.keys():
    # If needed, create subfolder
    label_subfolder = '\\'.join ( [ darknet_labels_path, single_img_filename.split('\\')[0] ] ) 
    if not os.path.exists ( label_subfolder ) and len(single_img_filename.split('\\'))>1:
        print ("Creating dir", label_subfolder)
        os.mkdir ( label_subfolder )

    # Create darknet labels file
    label_filename = '\\'.join ( [ darknet_labels_path, single_img_filename.replace('.JPEG', '.txt') ] )
    label_file = open( label_filename, 'w')

    # Place all bounding boxes inside the label file
    for single_bbox in bboxes[single_img_filename]:
        (classname, xmin, xmax, ymin, ymax, width, height) = (single_bbox[0], single_bbox[1], single_bbox[2], single_bbox[3], single_bbox[4], single_bbox[5], single_bbox[6] )
        bb = convert( (width,height), (xmin, xmax, ymin, ymax) )
        label_file.write( str(det_cats[classname][0]) + " " + " ".join([str(a) for a in bb]) + '\n')    
    
    label_file.close()

    # Put image file name into a file-names-file
    imagefilenames_file.write ( '\\'.join( [images_root, single_img_filename] ) + '\n' )

imagefilenames_file.close()

#for year, image_set in sets:
#    if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):
#        os.makedirs('VOCdevkit/VOC%s/labels/'%(year))
#    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
#    list_file = open('%s_%s.txt'%(year, image_set), 'w')
#    for image_id in image_ids:
#        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
#        convert_annotation(year, image_id)
#    list_file.close()


