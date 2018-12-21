# Draws confidences of predictions 
#   Have a model trained model loaded or load: 
#        
#
# To run:
#   cd C:\labs\KerasImagenetFruits\KerasTrainImagenet
#   python
#   exec(open("reimport.py").read())
#   model = load_model("D:\\ILSVRC14\\models\\model_v59.h5", custom_objects={'top_5':m_v11.top_5})
#   exec(open("visualPredConfidence.py").read())


# 1. Predict the entire dataset
# 2. Draw prediction intervals for each sample on a single line

#from DataGen import AugSequence_v4_PcaDistortion as as_v4
from Evaluation import Eval_v4_10framesaccuracy as e_v4
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

target_size = 224
#subtractMean = np.load("..\\rgb_mean.npy")
datasrc = "ilsvrc14_100boundingBoxes"

# Load category names
df = pd.read_excel("D:\\ILSVRC14\\ValidationFiles.xlsx", header=None)
class_indexes = df[0]
class_dirs = df[1]
class_names = df[2]

# 1. Predict the entire dataset (make sure shuffle=False so that predictions are in the same order as dataGen().classes
yhat, y = e_v4.eval(model, target_size=target_size, datasrc=datasrc, preprocess="vgg", shuffle=False, test=True, returnYhatY=True)

# Get the order (descending) of predictions
yhat_ord = np.fliplr ( np.argsort ( yhat, axis=1 ) )
# Get actual class
y_ord = np.argmax ( y, axis=1 )

# Sample indices for a single class 
for class_ind in range(6):
    #class_ind = 4                                           # make a loop for each class
    class_sample_ind = np.where (y_ord==class_ind)[0]       # samples of selected class
    m_class = len (class_sample_ind)                        # number of samples of selected class

    # Accumulate data for the plot to display top 5 predictions
    acc_y = np.zeros ( ( 0 ) )                              # acc_y: y-coordinate of the graph (same y corresponds to same sample)
    acc_width = np.zeros ( (0) )                            # acc_width: prediction confidence (width of the bar)
    acc_left = np.zeros ( (0) )                             # acc_left: sum of prediction confidences of higher-ranked classes of the sample
    acc_color = np.zeros ( (0), dtype='<U5' )               # green/red to indicate T/F; unicode <=5 characters
    prev_total_width = np.zeros ( (m_class) )               # prev_total_width: accumulated confidences for higher-ranked classes

    for pred_ind in range(5):
        acc_y = np.concatenate ( ( acc_y, np.arange(m_class) ) )
        this_pred_ind_width = yhat [class_sample_ind, yhat_ord [ class_sample_ind, pred_ind] ]
        acc_width= np.concatenate ( (acc_width,  this_pred_ind_width) )
        acc_left= np.concatenate ( (acc_left, prev_total_width) )
        prev_total_width += this_pred_ind_width
        acc_color= np.concatenate ( (acc_color, np.where ( yhat_ord[class_sample_ind, pred_ind]==y_ord[class_sample_ind] , "green", "red") ) )

    # Pick the label of highest incorrect class to display on the  y-axis
    tick_labels = np.zeros ( (m_class*5), dtype='<U20' )
    #Indexes of incorrect 1st or 2nd choice:
    incorrext_1st = np.where (yhat_ord[class_sample_ind,0] != y_ord[class_sample_ind])
    tick_labels [incorrext_1st] = "1." + class_names[ yhat_ord[class_sample_ind,0][incorrext_1st] ] 
    incorrext_2nd = np.where (yhat_ord[class_sample_ind,0] == y_ord[class_sample_ind])
    tick_labels [incorrext_2nd] = "2." + class_names[ yhat_ord[class_sample_ind,1][incorrext_2nd] ] 

    # Due to long text, remove secondary labels (after comma)
    tick_labels = [lst[0] for lst in np.core.defchararray.rsplit (tick_labels, ',')]

    # Set x axis width
    plt.xlim((0, 1))

    # Title is class label
    plt.title(class_names[class_ind])

    # 2. Draw prediction intervals for each sample on a single line
    mybar = plt.barh ( y=acc_y, \
        width= acc_width, \
        #height=0.8, \
        left=acc_left, \
        color=acc_color, \
        edgecolor="black" , \
        tick_label= tick_labels ) #y_ord [ class_sample_ind ]  )


    def onclick(event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        plt.scatter ( np.random.randint(0,10,5), np.random.randint(0,10,5))
        plt.show()

    #cid = plt.mpl_connect('button_press_event', onclick)
    mybar[0].figure.canvas.mpl_connect('button_press_event', onclick)
    plt.savefig ("..\\Visuals\\ConfidenceIntervals\\" + class_dirs[class_ind] + ".jpeg" )
    #plt.show ()