# Draws a confusion matrix of predicted vs. actual classes. Tested on 50 classes
#   Also, saves predicted vs. actual results for further analysis in ..\\pred_v202.csv
#   Have a model trained model loaded or load
#
# To run:
#   cd C:\labs\KerasImagenetFruits\KerasTrainImagenet
#   python
#   exec(open("reimport.py").read())
#   model = load_model("D:\\startup\\models\\model_v202_lessotherimages.h5", custom_objects={'top_5': m_v11.top_5})
#   exec(open("visualPredHeatmap_v202.py").read())

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

target_size = 224
datasrc = "sco_v2"

# RGB Mean over entire train set is saved by PCA.py
# subtractMean = np.load("..\\rgb_mean.npy")

#if vldDataGen is None:
vldDataGen = as_v5.AugSequence ( target_size=target_size, crop_range=1, allow_hor_flip=False, batch_size=32, \
    preprocess="vgg", datasrc=datasrc, test=True, shuffle=False )

#Confution Matrix and Classification Report
Y_pred = model.predict_generator ( vldDataGen, steps=len(vldDataGen) )
y_pred = np.argmax(Y_pred, axis=1)

conf_mat = confusion_matrix ( vldDataGen.dataGen().classes, y_pred )

#print('Classification Report')
#target_names = ['Cats', 'Dogs', 'Horse']
#print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

#fig = plt.figure( figsize = (5,5) )
fig, ax = plt.subplots()
fig.set_size_inches  (5,5)


#ax.set_xlabel = "Actual"
#ax.set_ylabel = "Predicted"
#ax.set_title = "Confusion matrix"
fig.text(0.5, 0.01, 'Predicted', ha='center', va='center')
fig.text(0.03, 0.5, 'Actual', ha='center', va='center', rotation='vertical')

im = ax.imshow(conf_mat)

# We want to show all ticks...
_ = ax.set_xticks(np.arange(conf_mat.shape[0]))
_ = ax.set_yticks(np.arange(conf_mat.shape[1]))

# Get barcodes (class names)
class_names = np.array(list( vldDataGen.dataGen().class_indices.keys() )) 

#Load real item names and show them on graph
df_itembarcodes = pd.read_csv ("D:\\Startup\\items.csv", header=None, dtype="str")

item_names = [ df_itembarcodes.loc [ df_itembarcodes[0] == class_name, 1 ].values[0] for class_name in class_names ]

_ = ax.set_xticklabels( item_names, rotation=45, ha ="right" )
_ = ax.set_yticklabels( item_names )

# ... and label them with the respective list entries
#ax.set_xticklabels(farmers)
#ax.set_yticklabels(vegetables)

# Rotate the tick labels and set their alignment.
#plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(conf_mat.shape[0]):
    for j in range(conf_mat.shape[1]):
        text = ax.text(j, i, conf_mat[i, j],
                       ha="center", va="center", color="w")

#ax.set_title("Harvest of local farmers (in tons/year)")
fig.tight_layout()
plt.show()

#conf_mat_nodiag = np.copy (conf_mat )
#for i in range(50):
#    conf_mat_nodiag[i,i]=0

df = pd.DataFrame()
df["filename"]=vldDataGen.dataGen().filenames
df["actual"]=vldDataGen.dataGen().classes
df["pred"]=y_pred
df.to_csv ("d:\\startup\\pred_v202.csv")
