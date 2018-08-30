# To run:
#   e_v2.eval(model), e_v2.eval(model, test=True)

from DataGen import DataGen_v1_150x150_1frame as dg_v1 
import math
import numpy as np

def eval ( model, test = False ):
    # Evaluates a given model's top 1-5 accuracy rate; prints result on screen
    #
    #   model: trained Keras model
    #

    trainDataGen = dg_v1.prepDataGen( test )

    #top 1,..5 error rates
    top_accuracy = np.zeros(5)

    #count of processed samples
    cnt_m = 0

    for x, y in trainDataGen:

        #number of samples in current minibatch
        m = x.shape[0]

        # Predict probabilities for each class
        pred = model.predict ( x )

        # Get order of predicted classes for all training examples (highest-prob classes - first)
        #  shape: [ m, c ]
        pred_class_order = np.flip ( np.argsort (pred, axis=1), axis=1 )

        # Get actual classes (reshape to [:,1] in order to be able to broadcast later)
        class_order = np.argmax (y, axis=1).reshape ( -1 , 1 )

        # Which_equal is [ m , c ] array - True is class matches actual class (1 true per row)
        which_equal = np.equal( pred_class_order, class_order.reshape(m,1) )

        # Calculate top 1-5 accuracy 
        for top in range( len( top_accuracy ) ):
            minibatch_accuracy = np.count_nonzero ( which_equal [:, 0:top+1] ) / float (m)
            top_accuracy[top] = (top_accuracy[top] * cnt_m + minibatch_accuracy * m ) / float ( cnt_m + m )

        cnt_m += m

        if ( cnt_m / trainDataGen.batch_size % 10 ) == 0:
            print ( "cnt_m, top_accuracy", str(cnt_m), top_accuracy )

        # ImageDataGenerator will loop forever
        if math.ceil ( cnt_m / trainDataGen.batch_size ) >= len(trainDataGen):
            break

    print ("top_accuracy",top_accuracy)
    #testDataGen = dg_v1.prepDataGen ( test=True )

    #return (ev_train, ev_test)
