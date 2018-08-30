# Trains a model for 50 epochs
#
# To run:
#   model = t_v22.trainModel()

from DataGen import DataGen_v2_150x150_shift_horflip as dg_v2
from Model import Model_v2_addDropout as m_v2

def trainModel():
    # Trains a model
    #
    # Returns: 
    #   model: trained Keras model

    dataGen = dg_v2.prepDataGen()

    model = m_v2.prepModel()

    epochs = 200

    model.fit_generator ( dataGen, steps_per_epoch=len(dataGen), epochs=epochs, verbose=1 )

    return model