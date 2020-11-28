# Given a model and a folder, make a confusion  matrix

from datetime import date

# result confusion matrix file
conf_mat_file = "D:\\Retellect\\errorAnalysis\\ConfMat_{}.png".format(date.today().strftime("%Y%m%d"))

# test images folder (split by true barcode)
test_folder = 'C:\\RetellectImages\\Test'

# model file
model_file = "A:\\RetellectModels\\model_20201121_59prekes.h5"

products_file = "D:\\Retellect\\data-prep\\temp\\prekes.selected.csv"

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# load products
df_products = pd.read_csv(products_file, header=None, names=["ProductName","Barcode","filesCnt"])

# load model
model = load_model( model_file )

# extract input size
target_size = model.layers[0].input_shape[1]

testDataGen = ImageDataGenerator( rescale=1./255 )
test_iterator = testDataGen.flow_from_directory(
        directory=test_folder,
        target_size=(target_size, target_size),
        batch_size=32,
        shuffle=False,
        class_mode='categorical')

# Predict highest classes
predictions = model.predict_generator(test_iterator, steps=len(test_iterator))
pred_classes = np.argmax(predictions, axis=1)

conf_mat = confusion_matrix(y_true=test_iterator.classes, y_pred=pred_classes)

# Draw confusion matrix
#sns.set(font_scale=3.0)
#ax = sns.heatmap(np.round(conf_mat / np.sum(conf_mat) * 100, decimals=1), annot=True, fmt='.1f', cbar=False)
ax = sns.heatmap(conf_mat, annot=True, cbar=False,annot_kws={'size':5})
#for t in ax.texts: t.set_text(t.get_text() + " %")

prods_short = [prod[0:10] for prod in df_products["ProductName"]]
#print (prods_short)

ax.set_xticks( np.arange(len(prods_short)) )
ax.set_yticks( np.arange(len(prods_short)) )
ax.set_yticklabels(prods_short , horizontalalignment='right', rotation = 0, size=5)
ax.set_xticklabels(prods_short , horizontalalignment='right', rotation = 90, size=5)  
ax.set_xlabel("PREDICTED", weight="bold")#, size=20)
ax.set_ylabel("ACTUAL", weight="bold")#, size=20)
plt.tight_layout()
plt.savefig(conf_mat_file)
plt.close()