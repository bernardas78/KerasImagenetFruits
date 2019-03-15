
# Modified Sample from https://towardsdatascience.com/yolo-object-detection-with-opencv-and-python-21e50ac599e9
#   Trained darknet with my own set of classes (imagenet)
#   Produces bounding boxes for 20 classes x 20 images of the validation set

# To run:
#   cd C:\labs\KerasImagenetFruits\Yolo
#   python visualPredDetYolo.py

args_image ="D:\\ILSVRC14\\darknet_data\\JPEGImages\\ILSVRC2014_DET_train_unp\\n02033041\\n02033041_5.JPEG" 
args_config = "C:\\labs\\darknet_imagenet\\yolov3_imagenet.cfg"
args_weights = "C:\\labs\\darknet\\build\\darknet\\x64\\backup\\yolov3_imagenet_last.weights" 
args_classes = "D:\\ILSVRC14\\darknet_data\\imagenet.names"

# How many classes to produce images with bounding boxes for
class_count = 2

# How many images for each class to produce
image_count = 200

#region part1
# import required packages
import cv2
import numpy as np
import os
#endregion

#region part2

# read class names from text file
classes = None
with open(args_classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# generate different colors for different classes 
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
net = cv2.dnn.readNet(args_weights, args_config)

#endregion

#region part3
# function to get the output layer names 
# in the architecture
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, int(confidence*8))

    cv2.putText(img, label + " {0:.2f}".format(confidence), (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#endregion


#region part4
def processImage(args_image, picDir, predVisualsDir):
    # read input image
    image = cv2.imread( "\\".join( [ picDir, args_image] ) )

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 1./255.

    # create input blob 
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    # set input blob for the network
    net.setInput(blob)

    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.2
    nms_threshold = 0.4

    # for each detetion from each output layer 
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    #print ("outs:", len(outs))
    for out in outs:
        #print ("detection:", out.shape)
        for detection in out:
            #print ('detection,out:', detection, out)
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                #print ("conf:", confidence)
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    #endregion

    #region part5
    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
    
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    # display output image    
    #cv2.imshow("object detection", image)

    # wait until any key is pressed
    #cv2.waitKey()
    
    # save output image to disk (if any bounding box found)
    if len(indices) > 0:
        cv2.imwrite( "\\".join( [ predVisualsDir, args_image] ), image)

    # release resources
    #cv2.destroyAllWindows()
    #endregion


# process some images for each some classes

#picDir = "D:\\ILSVRC14\\darknet_data\\JPEGImages\\ILSVRC2014_DET_train_unp\\"
picDir = "D:\\ILSVRC14\\darknet_data\\JPEGImages\\ILSVRC2013_DET_val\\all"
predVisualsDir = "C:\\labs\\KerasImagenetFruits\\Visuals\\PredYolo"

for _,_,files in os.walk(picDir):
    # Train pics in sub dirs
    #for subdir in subdirs:
    #    print ("Class ", subdir)
    for file in files:
        processImage(file, picDir, predVisualsDir)
        image_count -= 1
        if image_count<=0:
            break
