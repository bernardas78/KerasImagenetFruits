# Cuts longer dimension of a rectangular image to make it square and compresses to 256x256;
#   Saves resulting image to folder
#
# Running:
# cd C:\labs\KerasImagenetFruits\PreprocessImages
# python
#
# exec(open("Squarize.py").read())

from PIL import Image
#from imgaug import augmenters as iaa
import numpy as np
import os
import time

#batchSize = 100         # How many images are loaded at a time
#angleCntPerCircle = 8   # Number of eqyal angles to divide a circle to for each image
#edge = 50               # Edge length of a compressed image

#picDir = "C:\\ILSVRC14\\ILSVRC2012_img_train_unp_100.NonSquare"
#squarePicDir = "C:\\ILSVRC14\\ILSVRC2012_img_train_unp_100"
picDir = "D:\\ILSVRC14\\ILSVRC2012_bbox"
squarePicDir = "D:\ILSVRC14\ILSVRC2012_bbox.Square255"
ind=0

dirNames = []
rotatedDirNames = []
fileNames = []
ind=0

print ("Looping through file names")
for _,dirs,files in os.walk(picDir):
    #print("files.shape:",len(files))
    for dir in dirs:
        #print("subdir,dir:",os.path.join(subdir,dir))
        keywordDir = os.path.join(picDir,dir)
        squareKeywordDir = os.path.join(squarePicDir,dir)
        if not os.path.exists(squareKeywordDir):
            os.mkdir(squareKeywordDir)
        for _,_,files in os.walk(keywordDir):
            for file in files:
                ind+=1

                fileName = os.path.join(keywordDir,file)

                try:
                    im = Image.open( fileName )
                    #im = 
                    h = np.asarray (im).shape[0]
                    w = np.asarray (im).shape[1]

                    if h<w:
                        im = im.crop(( int((w-h)/2), 0, int((w-h)/2)+h , h))
                    else:
                        im = im.crop(( 0, int((h-w)/2), w, int((h-w)/2)+w ))

                    im = im.resize ( (255,255) )
                    squareFileName = os.path.join (squareKeywordDir, file)
                    im.save( squareFileName )
                except:
                    print ("Exception processing file ", fileName)
                #dirNames.append (keywordDir)
                #rotatedDirNames.append (rotatedKeywordDir)
                #fileNames.append(file)
                #fileName = os.path.join(keywordDir,file)
                #fileNames.append(fileName)

                #rotatedFileName = os.path.join(rotatedKeywordDir,file)
                #rotatedFileNames.append(rotatedFileName)

                if np.mod(ind,1000)==0:
                    print (str(ind),"files added")
#seqs = []
#for angle_ind in range(angleCntPerCircle):
#    angle = angle_ind * 360/angleCntPerCircle
#    seq = iaa.Sequential([
#        iaa.Grayscale(alpha=1.),                    # 1 for grayscale; 0 for full color
#        iaa.Scale({"height": edge, "width": edge}), # compresses the image to given dimensions
#        iaa.Affine(rotate=angle)                    #
#    ])
#    seqs.append(seq)

#for batch_id in range(int(np.ceil(len(fileNames)/batchSize))):
#    # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
#    # or a list of 3D numpy arrays, each having shape (height, width, channels).
#    # Grayscale images must have shape (height, width, 1) each.
#    # All images must have numpy's dtype uint8. Values are expected to be in
#    # range 0-255.
#    #images = load_batch(batch_idx)

#    images = []

#    lowInd = batchSize*batch_id
#    highInd = np.minimum ( batchSize*(batch_id+1), len(fileNames))

#    print ("Loading",lowInd,"-",highInd,"images")
#    start = time.perf_counter()
#    for img_ind in range(lowInd, highInd):
#        # Load image from file
#        img_file = os.path.join ( dirNames[img_ind], fileNames[img_ind] )
#        img = np.array(Image.open( img_file ))
        
#        # Make image square in order to preserve ratio when compressing
#        startBiggerDim = int(np.abs(image.shape[0] - image.shape[1]) / 2)
#        if img.shape[0] > img.shape[1]:
#            img = img[ startBiggerDim : startBiggerDim + img.shape[1] , : , ... ]
#        else:
#            img = img[ : , startBiggerDim : startBiggerDim + img.shape[0] , ... ]

#        # For single color images - change to 3 channels so that imagaug.Grayscale does not crash
#        if len(img.shape)==2:
#            img = np.broadcast_to(img[..., np.newaxis], (img.shape[0], img.shape[1], 3))

#        # For images having <3 channels - broadcast the first channel
#        if img.shape[2]<3:
#            img = np.broadcast_to(img[:,:,0][..., np.newaxis], (img.shape[0], img.shape[1], 3))

#        # Add to images array
#        images.append(img)
#        #print ("fileNames[img_ind],img.shape:",fileNames[img_ind],img.shape)
#    print (time.perf_counter() - start)


#    for angle_ind in range(angleCntPerCircle):
#        print ("Augmenting",highInd-lowInd,"images by", angle_ind * 360/angleCntPerCircle)

#        # Run the augmenter
#        start = time.perf_counter()
#        images_aug = seqs[angle_ind].augment_images(images)

#        print (time.perf_counter() - start)

#        print ("Saving",highInd-lowInd,"images")
#        start = time.perf_counter()

#        #Save images to files
#        for img_ind in range(lowInd,highInd):
#            imgAugmented = Image.fromarray(images_aug[img_ind-lowInd])
#            imgAugmented_file = os.path.join ( rotatedDirNames[img_ind] ,str(angle_ind)+"."+fileNames[img_ind] )
#            try:
#                imgAugmented.save(imgAugmented_file)
#            except:
#                print ("Failed to save:",imgAugmented_file)

#        print (time.perf_counter() - start)

#    #print ("image.shape:",image.shape)
#    #images = np.array([image, image2])
#    #images = [image, image2]
#    #print ("images.shape:",images.shape)

#    #j = Image.fromarray(images_aug[0])
#    #j2 = Image.fromarray(images_aug[1])
#    #j.save("apple1.png")
#    #j2.save("other.png")
