# Draws a confusion matrix of predicted vs. actual classes. Tested on 50 classes
#   Have a model trained model loaded or load: model = load_model("D:\\ILSVRC14\\models\\model_v53.h5") 
#
# To run:
#   cd C:\labs\KerasImagenetFruits\KerasTrainImagenet
#   python
#   exec(open("reimport.py").read())
#   exec(open("visualPredHeatmap.py").read())

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

target_size = 224
datasrc = "ilsvrc14_50classes"

# RGB Mean over entire train set is saved by PCA.py
subtractMean = np.load("..\\rgb_mean.npy")

if vldDataGen is None:
    vldDataGen = as_v3.AugSequence ( target_size=target_size, crop_range=1, allow_hor_flip=False, \
        batch_size=128, subtractMean=subtractMean, datasrc=datasrc, test=True, shuffle=False )

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
im = ax.imshow(conf_mat)

# We want to show all ticks...
_ = ax.set_xticks(np.arange(conf_mat.shape[0]))
_ = ax.set_yticks(np.arange(conf_mat.shape[1]))
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

df = pandas.DataFrame()
df["filename"]=vldDataGen.dataGen().filenames
df["actual"]=vldDataGen.dataGen().classes
df["pred"]=y_pred
df.to_csv ("..\\pred_v53.csv")
