# Trains a model for 50 epochs
#
# To run:
#   model = t_v4.trainModel()

from DataGen import DataGen_v1_150x150_1frame as dg_v1 
from Model import Model_v4_inception_trainbase as m_v4

def trainModel():
    # Trains a model
    #
    # Returns: 
    #   model: trained Keras model

    dataGen = dg_v1.prepDataGen()

    model = m_v4.prepModel()

    epochs = 50

    model.fit_generator ( dataGen, steps_per_epoch=len(dataGen), epochs=epochs, verbose=1 )

    return model