# To run:
#   e_v1.eval(model)

from DataGen import DataGen_v1_150x150_1frame as dg_v1 

def eval ( model, target_size=150, datasrc = "selfCreatedGoogle" ):
    # Evaluates a given model against train and test sets; prints result on screen
    #
    #   model: trained Keras model
    #
    # Returns: 
    #   [Loss, accuracy] for [Train, Validation] sets

    trainDataGen = dg_v1.prepDataGen( target_size=target_size, datasrc = datasrc )

    ev_train = model.evaluate_generator ( trainDataGen, verbose=1, steps=len(trainDataGen))
    #ev_train = model.evaluate_generator ( trainDataGen, verbose=1, steps=1)

    testDataGen = dg_v1.prepDataGen ( target_size=target_size, datasrc = datasrc, test=True )

    ev_test = model.evaluate_generator ( testDataGen, verbose=1, steps=len(testDataGen))

    return (ev_train, ev_test)
