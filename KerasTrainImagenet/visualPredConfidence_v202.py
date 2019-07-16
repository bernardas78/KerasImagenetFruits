# Draws confidences of predictions 
#   Have confidence-to-accuracy dependency, as well as a couple of thresholds
#        
#
# To run:
#   cd C:\labs\KerasImagenetFruits\KerasTrainImagenet
#   python
#   exec(open("reimport.py").read())
#   model = load_model("D:\\Startup\\models\\model_v202.h5", custom_objects={'top_5':m_v11.top_5})
#   Set threshold 0, 200, 300 (0%, 20%, 30%)
threshold = 630
#   exec(open("visualPredConfidence_v202.py").read())

# 1. Predict the entire dataset
# 2. Draw prediction intervals for each sample on a single line

#from DataGen import AugSequence_v4_PcaDistortion as as_v4
from Evaluation import Eval_v4_10framesaccuracy as e_v4
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

target_size = 224
#subtractMean = np.load("..\\rgb_mean.npy")
datasrc = "sco_v2"

# Load category names
#df = pd.read_excel("D:\\ILSVRC14\\ValidationFiles.xlsx", header=None)
#class_indexes = df[0]
#class_dirs = df[1]
#class_names = df[2]

# 1. Predict the entire dataset (make sure shuffle=False so that predictions are in the same order as dataGen().classes
yhat, y = e_v4.eval(model, target_size=target_size, datasrc=datasrc, preprocess="vgg", batch_size=48, shuffle=False, test=True, returnYhatY=True)

# Get sorted (desc) prediction values (i.e. confidence of being right)
yhat_sorted = np.fliplr ( np.sort ( yhat, axis=1 ) )
# Get the order (descending) of predictions
yhat_ord = np.fliplr ( np.argsort ( yhat, axis=1 ) )
# Get actual class
y_ord = np.argmax ( y, axis=1 )
# For numerical stability
epsilon = 0.001

# Show up to 5 items based on confidence
for top_k in range(1):
    # Is correct class in the top_k predictions? [m,], True/False
    top_k_incl_correct = np.equal(yhat_ord [ :, 0:(top_k+1) ], np.expand_dims(y_ord,axis=1)).any( axis=1 )
    # Confidence of being right is the sum of top_k predictions; size [m,]
    conf = np.sum ( yhat_sorted [ :, 0:(top_k+1) ], axis=1)
    # Set graph-points:
    #   X-axis is threshold 0-100% with 0.1% increments
    #   Y-axis is accuracy as a result of using a threshold (0% threshold constitutes to top_k accuracy)
    xdata, acc, total_served, well_served = (np.arange(0., 1.0, 0.001), np.zeros(1000), np.zeros(1000), np.zeros(1000))
    for pct_ind in range(1000):
        # Threshold: mimnimum confidence that allows to show top_k items
        conf_threshold = pct_ind / 1000.
        # Wheather samples that pass the minimum accuracy requirement (these will show top_k; others - full tree)
        pass_threshold = conf >= conf_threshold
        # What's the accuracy of only those samples that pass required threshold
        acc[pct_ind] = len(np.where(pass_threshold & top_k_incl_correct)[0]) / (len(np.where(pass_threshold)[0]) + epsilon)
        # Total percentage of customers server (those shown top_k)
        total_served[pct_ind] = len(np.where(pass_threshold)[0]) / len(pass_threshold)
        # Well served percentage = total served * accuracy
        well_served[pct_ind] = total_served[pct_ind] * acc[pct_ind]

        # Let's see what we gain by putting a 20%, 30% confidence threshold
        if pct_ind == threshold:
            # Threshold line and label
            plt.plot ( [pct_ind/1000., pct_ind/1000.], [0., 1.], color='red')
            plt.text ( pct_ind/1000., 0., "Threshold: %.2f" % float(pct_ind/1000.) , horizontalalignment='right', verticalalignment='bottom', rotation=90, color='red')
            # Accuracy line and label for 30% threshold
            plt.plot ( [0., 1.], [acc[pct_ind], acc[pct_ind]], color='blue')
            plt.text ( pct_ind/1000., acc[pct_ind], "Accuracy: %.2f" % acc[pct_ind], horizontalalignment='right', verticalalignment='bottom', color='blue' )
            # Total served line and label for 30% threshold
            plt.plot ( [0., 1.], [total_served[pct_ind], total_served[pct_ind]], color='limegreen')
            plt.text ( 0.49, total_served[pct_ind]+.01, "Total served: %.2f" % total_served[pct_ind], horizontalalignment='right', color='limegreen' )
            # Well served line and label for 30% threshold
            plt.plot ( [0., 1.], [well_served[pct_ind], well_served[pct_ind]], color='lime')
            plt.text ( 0.49, well_served[pct_ind]-.01, "Well served: %.2f" % well_served[pct_ind], horizontalalignment='right', verticalalignment='top', color='lime' )

    # Accuracy, total customers served, customers correctly served lines
    plt.plot ( xdata, acc, color='blue' )
    plt.plot ( xdata, total_served, color='limegreen')
    plt.plot ( xdata, well_served, color='lime')
    if threshold > 0:
        plt.title ( "Impact on accuracy of putting a %.2f confidence threshold" % float(threshold/1000.) )
    else:
        plt.title ("Accuracy with no threshold")
plt.xlim((0., 0.8))
plt.ylim((0., 1.05))
#plt.show()
plt.savefig ("D:\\Startup\\ConfThreshold" + str(threshold) + ".jpg")
# Sample indices for a single class 
#for class_ind in range(6):
#    #class_ind = 4                                           # make a loop for each class
#    class_sample_ind = np.where (y_ord==class_ind)[0]       # samples of selected class
#    m_class = len (class_sample_ind)                        # number of samples of selected class

#    # Accumulate data for the plot to display top 5 predictions
#    acc_y = np.zeros ( ( 0 ) )                              # acc_y: y-coordinate of the graph (same y corresponds to same sample)
#    acc_width = np.zeros ( (0) )                            # acc_width: prediction confidence (width of the bar)
#    acc_left = np.zeros ( (0) )                             # acc_left: sum of prediction confidences of higher-ranked classes of the sample
#    acc_color = np.zeros ( (0), dtype='<U5' )               # green/red to indicate T/F; unicode <=5 characters
#    prev_total_width = np.zeros ( (m_class) )               # prev_total_width: accumulated confidences for higher-ranked classes

#    for pred_ind in range(5):
#        acc_y = np.concatenate ( ( acc_y, np.arange(m_class) ) )
#        this_pred_ind_width = yhat [class_sample_ind, yhat_ord [ class_sample_ind, pred_ind] ]
#        acc_width= np.concatenate ( (acc_width,  this_pred_ind_width) )
#        acc_left= np.concatenate ( (acc_left, prev_total_width) )
#        prev_total_width += this_pred_ind_width
#        acc_color= np.concatenate ( (acc_color, np.where ( yhat_ord[class_sample_ind, pred_ind]==y_ord[class_sample_ind] , "green", "red") ) )

#    # Pick the label of highest incorrect class to display on the  y-axis
#    tick_labels = np.zeros ( (m_class*5), dtype='<U20' )
#    #Indexes of incorrect 1st or 2nd choice:
#    incorrext_1st = np.where (yhat_ord[class_sample_ind,0] != y_ord[class_sample_ind])
#    tick_labels [incorrext_1st] = "1." + class_names[ yhat_ord[class_sample_ind,0][incorrext_1st] ] 
#    incorrext_2nd = np.where (yhat_ord[class_sample_ind,0] == y_ord[class_sample_ind])
#    tick_labels [incorrext_2nd] = "2." + class_names[ yhat_ord[class_sample_ind,1][incorrext_2nd] ] 

#    # Due to long text, remove secondary labels (after comma)
#    tick_labels = [lst[0] for lst in np.core.defchararray.rsplit (tick_labels, ',')]

#    # Set x axis width
#    plt.xlim((0, 1))

#    # Title is class label
#    plt.title(class_names[class_ind])

#    # 2. Draw prediction intervals for each sample on a single line
#    mybar = plt.barh ( y=acc_y, \
#        width= acc_width, \
#        #height=0.8, \
#        left=acc_left, \
#        color=acc_color, \
#        edgecolor="black" , \
#        tick_label= tick_labels ) #y_ord [ class_sample_ind ]  )


#    def onclick(event):
#        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
#              ('double' if event.dblclick else 'single', event.button,
#               event.x, event.y, event.xdata, event.ydata))
#        plt.scatter ( np.random.randint(0,10,5), np.random.randint(0,10,5))
#        plt.show()

#    #cid = plt.mpl_connect('button_press_event', onclick)
#    mybar[0].figure.canvas.mpl_connect('button_press_event', onclick)
#    plt.savefig ("..\\Visuals\\ConfidenceIntervals\\" + class_dirs[class_ind] + ".jpeg" )
#    plt.show ()