#   Presumed: 
#       have a (trained) model
#   Print: 
#       activations of each layer summaries for a single training example (change batch_size toprint of more)
#       weights, biases summaries of every layer in a trained model
#       gradients of model given a single training example
# To run:
#   cd C:\labs\KerasImagenetFruits\KerasTrainImagenet
#   python
#   exec(open("diagnostics.py").read())

from keras.backend import function
from keras import backend as K
import tensorflow as tf
import numpy as np

dataGen = dg_v1.prepDataGen ( target_size=224, batch_size=1, datasrc="ilsvrc14_50classes", test=False)
for X, y in dataGen:
    break


print ("ACTIVATIONS SUMMARY")
for layerind in range ( len (model.layers) ):
    layer = model.layers [ layerind ]

    func_activation = function ( [ model.input ], [ model.layers [ layerind ].output ] )

    output_activation = func_activation ( [ X ] ) [0] 

    minval = np.min (output_activation)
    maxval = np.max (output_activation)
    meanval = np.mean (output_activation)
    stdval = np.std (output_activation)

    flatarr = np.ravel ( output_activation )
    overzeropct = len ( flatarr [ np.where ( flatarr>1e-9 ) ] ) / len(flatarr) * 100.
    bellowzeropct = len ( flatarr [ np.where ( flatarr<-1e-9 ) ] ) / len(flatarr) * 100.
    zeropct = 100. - overzeropct - bellowzeropct

    print ("Std: %8.5f; mean: %8.5f; <=>0: [%5.1f %5.1f %5.1f]" % (stdval, meanval, bellowzeropct, zeropct, overzeropct ), "---",layer.name, output_activation.shape )

print (" ")
print ("WEIGHTS SUMMARY")
for weightind in range ( len ( model.get_weights() ) ):
    wt = model.get_weights()[weightind]

    minval = np.min (wt)
    maxval = np.max (wt)
    meanval = np.mean (wt)
    stdval = np.std (wt)

    flatarr = np.ravel ( wt )
    overzeropct = len ( flatarr [ np.where ( flatarr>1e-9 ) ] ) / len(flatarr) * 100.
    bellowzeropct = len ( flatarr [ np.where ( flatarr<-1e-9 ) ] ) / len(flatarr) * 100.
    zeropct = 100. - overzeropct - bellowzeropct

    print ("Std: %8.5f; mean: %8.5f; <=>0: [%5.1f %5.1f %5.1f]; Index: %2d;" % (stdval, meanval, bellowzeropct, zeropct, overzeropct, weightind), "Shape:", wt.shape )

print (" ")
print ("GRADIENTS SUMMARY")

# Calculate gradients
loss = K.sum ( K.categorical_crossentropy ( y, model.output) )
gradients = K.gradients ( loss, model.trainable_weights ) 
sess = K.get_session()
evaluated_gradients = sess.run ( gradients, feed_dict = {model.input:X} )

for gradind in range ( len ( evaluated_gradients ) ):
    grad = evaluated_gradients[gradind]

    minval = np.min (grad)
    maxval = np.max (grad)
    meanval = np.mean (grad)
    stdval = np.std (grad)

    flatarr = np.ravel ( grad )
    overzeropct = len ( flatarr [ np.where ( flatarr>1e-9 ) ] ) / len(flatarr) * 100.
    bellowzeropct = len ( flatarr [ np.where ( flatarr<-1e-9 ) ] ) / len(flatarr) * 100.
    zeropct = 100. - overzeropct - bellowzeropct

    print ("Std: %11.8f; mean: %11.8f; <=>0: [%5.1f %5.1f %5.1f]; Index: %2d;" % (stdval, meanval, bellowzeropct, zeropct, overzeropct, gradind), "Shape:", grad.shape )

