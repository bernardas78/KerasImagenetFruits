# Train a saved model a bit longer to see if it produces better results 
#   while training in iterations 150->200, it showed improvement of 41.4->46.7%, so should be promissing

from keras.models import load_model
from DataGen import DataGen_v2_150x150_shift_horflip as dg_v2

model = load_model("C:\\labs\\models\\model_v22.h5")

dataGen = dg_v2.prepDataGen()

epochs = 200

model.fit_generator ( dataGen, steps_per_epoch=len(dataGen), epochs=epochs, verbose=1 )
