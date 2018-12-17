# Draws confidences of predictions 
#   Have a model trained model loaded or load: 
#        
#
# To run:
#   cd C:\labs\KerasImagenetFruits\KerasTrainImagenet
#   python
#   exec(open("reimport.py").read())
#   model = load_model("D:\\ILSVRC14\\models\\model_v56.h5", custom_objects={'top_5':m_v8.top_5})
#   exec(open("visualPredConfidence.py").read())


# 1. Predict the entire dataset
# 2. Draw prediction intervals for each sample on a single line

#from DataGen import AugSequence_v4_PcaDistortion as as_v4
from Evaluation import Eval_v4_10framesaccuracy as e_v4
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

target_size = 224
subtractMean = np.load("..\\rgb_mean.npy")
datasrc = "ilsvrc14_100boundingBoxes"

# Load category names
df = pd.read_excel("D:\\ILSVRC14\\ValidationFiles.xlsx", header=None)
class_indexes = df[0]
class_dirs = df[1]
class_names = df[2]

# 1. Predict the entire dataset (make sure shuffle=False so that predictions are in the same order as dataGen().classes
#yhat, y = e_v4.eval(model, target_size=target_size, subtractMean=subtractMean, datasrc=datasrc, test=True, returnYhatY=True)

# Get the order (descending) of predictions
yhat_ord = np.fliplr ( np.argsort ( yhat, axis=1 ) )
# Get actual class
y_ord = np.argmax ( y, axis=1 )

# Sample indices for a single class (=0)
class_ind = 3
class_sample_ind = np.where (y_ord==class_ind)[0]
m_class = len (class_sample_ind)

# Accumulate data for the plot to display top 5 predictions
acc_y = np.zeros ( ( 0 ) )
acc_width = np.zeros ( (0) )
acc_left = np.zeros ( (0) )
acc_color = np.zeros ( (0), dtype='<U5' ) # unicode <=5 characters

prev_total_width = np.zeros ( (m_class) )

for pred_ind in range(5):
    acc_y = np.concatenate ( ( acc_y, np.arange(m_class) ) )
    this_pred_ind_width = yhat [class_sample_ind, yhat_ord [ class_sample_ind, pred_ind] ]
    acc_width= np.concatenate ( (acc_width,  this_pred_ind_width) )
    acc_left= np.concatenate ( (acc_left, prev_total_width) )
    prev_total_width += this_pred_ind_width
    acc_color= np.concatenate ( (acc_color, np.where ( yhat_ord[class_sample_ind, pred_ind]==y_ord[class_sample_ind] , "green", "red") ) )

# 2. Draw prediction intervals for each sample on a single line
plt.barh ( y=acc_y, \
    width= acc_width, \
    #height=0.8, \
    left=acc_left, \
    color=acc_color, \
    edgecolor="black" , \
    tick_label="" ) #y_ord [ class_sample_ind ]  )
#plt.gca().set_yticks(["bla","bla"])

plt.show ()