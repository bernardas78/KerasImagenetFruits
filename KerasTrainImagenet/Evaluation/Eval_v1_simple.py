# To run:
#   e_v1.eval(model)

from DataGen import DataGen_v1_150x150_1frame as dg_v1 

def eval ( model ):
    # Evaluates a given model against train and test sets; prints result on screen
    #
    #   model: trained Keras model
    #
    # Returns: 
    #   [Loss, accuracy] for [Train, Validation] sets

    trainDataGen = dg_v1.prepDataGen()

    ev_train = model.evaluate_generator ( trainDataGen, verbose=1)

    testDataGen = dg_v1.prepDataGen ( test=True )

    ev_test = model.evaluate_generator ( testDataGen, verbose=1)

    return (ev_train, ev_test)
