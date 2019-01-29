# Make 2 dictionarries:
#   DET train cat: DET vld cat (vld categories are higher or same level as train categories)
#   DET det_cat_name:[det_cat_ind, description] (used for encoding labels)
# To load:
#   import pickle
#   det_cats = pickle.load( open('d:\ILSVRC14\det_catdesc.obj', 'rb') )
#   det_cat_hier = pickle.load( open('d:\ILSVRC14\det_cathier.obj', 'rb') )
# To run:
#   cd C:\labs\KerasImagenetFruits\PreprocessImages
#   python
#   exec(open("imagenetCatHierToDictToFile.py").read())

import scipy.io
import random
import pickle

# Dictionary of category hierarchies
cat_hier = {}

# Hierarchical dataset of imagenet categories (part of devkit)
det_categories_file = 'D:\ILSVRC14\ILSVRC2014_devkit\data\meta_det.mat'

# DET validation data - actual categories used in DET challenge
bboxDir = "d:\\ILSVRC14\\ILSVRC2013_DET_bbox_val\\"
picDir = "d:\\ILSVRC14\\ILSVRC2013_DET_val\\"

# Target file name for DET category hierarchy and descriptions
det_cat_desc_file = 'd:\ILSVRC14\det_catdesc.obj'
det_cat_hier_file = 'd:\ILSVRC14\det_cathier.obj'

cats =  scipy.io.loadmat( det_categories_file )

# First list of all categories and dictioanry name:description (count 815)
cat_names = [cat[1][0] for cat in cats['synsets'][0]]

# Dictionary of cat_name:description
cat_desc = {}
for cat in cats['synsets'][0]:
    cat_desc [cat[1][0]] = [cat[2][0]]

# fill dictionary child:parent
for single_cat in cats['synsets'][0]:

    # Make sure only 1 element in names
    assert ( len ( single_cat [ 1 ] ) == 1 )
    cat_name = single_cat [ 1 ][0]
    if len ( single_cat [ 4 ] ) > 0:
        cat_children = single_cat [ 4 ][0]
        for cat_child in cat_children:
            cat_hier[ cat_names[cat_child-1] ] = cat_name

# Print a full list of child : parent
for cat in cat_hier.keys():
    print( cat_desc[cat], ':', cat_desc[cat_hier[cat]] )

# Produce a detection challenge hierarchy:
#   Validation only contain upper level categories (not necesarily root)
#   Training contains lower level categories too
#   Dictionary will contain lower:upper, where:
#       upper - only DET challenge categories
#       lower - all categories from imagenet except higher-than-DET categories

# Pascal VOC format parser for a single folder
exec(open("voc\\voc.py").read())

# Load list of DET categories 
cache_name = "cache_" + str(random.randint (0,1000000))
ann = parse_voc_annotation (bboxDir, picDir, cache_name)
print ("Loaded DET categories")

# remove junk left by parse_voc_annotation() 
os.remove(cache_name)

# Detection categories only (count 200)
#det_cats = ann[1].keys()

# Dictionary of det_cat_name:[det_cat_ind, description]
det_cats = {}
det_cat_index = 0
for det_cat_name in ann[1].keys():
    det_cats [det_cat_name] = [ det_cat_index, cat_desc[det_cat_name][0] ]
    det_cat_index +=1



det_cat_hier = {} 

# Dictionary of train_cat:val_cat
for lower_cat in cat_names:
    # Finding parent (or grand parent) category
    parent_found = False
    candidate_parent = lower_cat
    #print ("Looking for parent of ", cat_desc [lower_cat])
    while not parent_found:
        if candidate_parent in det_cats.keys():
            parent_found=True
            det_cat_hier [lower_cat] = candidate_parent
            #print ("Found parent of ", cat_desc [lower_cat], " is ", cat_desc [candidate_parent])
        elif candidate_parent in cat_hier.keys():
            candidate_parent = cat_hier [candidate_parent]
            #print ("Next candidate parent is ", cat_desc [candidate_parent])
        else:
            # Category not found under on of DET candidates; don't add to dict
            #print ("Parent of ", cat_desc [lower_cat], " not found")
            break
            
print  ("DET CHALLENGE SUBCATEGORIES:CATEGORIES")
for key in det_cat_hier.keys():
    print (cat_desc[key],":",cat_desc[det_cat_hier[key]])
print ("total sub-cats, cats:", len(det_cat_hier), len(det_cats.keys()))

print ( "Save category descriptions to a single file" )
with open(det_cat_desc_file, 'wb') as file_desc:
    pickle.dump(det_cats, file_desc)

print ( "Save det category hierarchy to a single file" )
with open(det_cat_hier_file, 'wb') as file_hier:
    pickle.dump(det_cat_hier, file_hier)

